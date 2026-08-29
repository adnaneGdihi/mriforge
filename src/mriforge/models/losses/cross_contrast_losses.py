import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.losses.registry import register_loss


@register_loss(name="lncc", aliases=["LNCCLoss", "local_ncc"])
class LNCCLoss(nn.Module):
    """Localized Normalized Cross-Correlation (LNCC) Loss.

    Optimizes multi-contrast/cross-scanner images using local structural patches,
    making the loss analytically invariant to global and local monotonic intensity shifts
    caused by different scanner field strengths (T1/T2 variations).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{LNCC} = - \frac{\text{Cov}(\\hat{y}, y)^2}{\text{Var}(\\hat{y}) \text{Var}(y) + \\epsilon}"""

    def __init__(self, window_size: int = 9, dims: int = 2, eps: float = 1e-8):
        """__init__.

        Args:
            window_size (int): Description.
            dims (int): Description.
            eps (float): Description.
        """
        super().__init__()
        self.window_size = window_size
        self.dims = dims
        self.eps = eps

        # Precompute the convolution kernel of all ones
        if dims == 2:
            self.register_buffer("kernel", torch.ones(1, 1, window_size, window_size))
            self.conv = F.conv2d
            self.pad = tuple([window_size // 2] * 4)
        elif dims == 3:
            self.register_buffer("kernel", torch.ones(1, 1, window_size, window_size, window_size))
            self.conv = F.conv3d
            self.pad = tuple([window_size // 2] * 6)
        else:
            raise ValueError(f"LNCC only supports 2D and 3D. Got {dims}D.")

        self.window_num_elements = window_size**dims

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative LNCC (since we want to minimize the loss).

        Args:
            pred: Reconstructed magnitude image [B, C, H, W] or [B, C, D, H, W]
            target: Target magnitude image
        Returns:
            Scalar LNCC loss (negative, so minimization maximizes correlation)

        forward method for LNCCLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure single channel representations for structure check
        if pred.shape[1] > 1:
            pred = torch.mean(pred, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)

        # Pad to keep output same size
        pred_pad = F.pad(pred, self.pad, mode="reflect")
        target_pad = F.pad(target, self.pad, mode="reflect")

        # Local means
        mu_p = self.conv(pred_pad, self.kernel.to(pred.device)) / self.window_num_elements
        mu_t = self.conv(target_pad, self.kernel.to(target.device)) / self.window_num_elements

        # Zero-mean patches
        pred_zm = pred - mu_p
        target_zm = target - mu_t

        pred_zm_pad = F.pad(pred_zm, self.pad, mode="reflect")
        target_zm_pad = F.pad(target_zm, self.pad, mode="reflect")

        # Local variances and covariance
        # Var(P)
        p_sq = pred_zm_pad**2
        var_p = self.conv(p_sq, self.kernel.to(pred.device))

        # Var(T)
        t_sq = target_zm_pad**2
        var_t = self.conv(t_sq, self.kernel.to(target.device))

        # Cov(P, T)
        pt = pred_zm_pad * target_zm_pad
        cov_pt = self.conv(pt, self.kernel.to(pred.device))

        # LNCC
        lncc = (cov_pt**2) / (var_p * var_t + self.eps)

        # We negate it for minimization. Averaged across all voxels
        return -torch.mean(lncc)
