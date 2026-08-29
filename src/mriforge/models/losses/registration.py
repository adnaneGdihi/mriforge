import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.losses.registry import register_loss


# `lncc` alias intentionally dropped — it was colliding with the canonical
# `lncc` registration in `cross_contrast_losses.py` (a different LNCC
# tailored to multi-contrast registration). The canonical disambiguated
# handle for *this* implementation is `local_cross_correlation_loss`. See
# TODO/audit/12_losses.md F2.
@register_loss("local_cross_correlation_loss", aliases=["lncc_loss"])
class LocalCrossCorrelationLoss(nn.Module):
    """Computes Local Normalized Cross Correlation (LNCC).
    Robust to intensity variations between scanners.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{LCC} = - \\mathbb{E} \\left[ \frac{\text{Cov}(\\hat{y}, y)^2}{\text{Var}(\\hat{y}) \text{Var}(y) + \\epsilon} \right]"""

    def __init__(self, kernel_size=9):
        """__init__.

        Args:
            kernel_size (Any): Description.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, I, J):
        # I: Corrected 64mT, J: Target 3T
        # Window filtering
        """forward.

        Args:
            I (Any): Description.
            J (Any): Description.
        Returns:
            Any: Description.

        forward method for LocalCrossCorrelationLoss.

        Executes PyTorch tensor operations.

        Args:
            I (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            J (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        win = torch.ones(1, 1, self.kernel_size, self.kernel_size).to(I.device) / (
            self.kernel_size**2
        )

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, win, padding=self.padding)
        J_sum = F.conv2d(J, win, padding=self.padding)
        I2_sum = F.conv2d(I2, win, padding=self.padding)
        J2_sum = F.conv2d(J2, win, padding=self.padding)
        IJ_sum = F.conv2d(IJ, win, padding=self.padding)

        cross = IJ_sum - I_sum * J_sum
        # Var = E[X^2] - E[X]^2 is non-negative analytically, but can dip a few
        # ULP below zero from float round-off. Clamp so the denominator cannot
        # flip sign (a negative I_var with positive J_var would make the loss
        # explode / go NaN near flat regions).
        I_var = (I2_sum - I_sum * I_sum).clamp_min(0.0)
        J_var = (J2_sum - J_sum * J_sum).clamp_min(0.0)

        cc = cross * cross / (I_var * J_var + 1e-5)
        return -1.0 * torch.mean(cc)


@register_loss("smoothness_loss", aliases=["smooth_loss", "flow_smoothness"])
class SmoothnessLoss(nn.Module):
    """Penalizes jagged deformations to ensure physically realistic B0 maps.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Smoothness} = \frac{1}{2} (\\mathbb{E}[|\nabla_x v|] + \\mathbb{E}[|\nabla_y v|])"""

    def forward(self, flow):
        """forward.

        Args:
            flow (Any): Description.
        Returns:
            Any: Description.

        forward method for SmoothnessLoss.

        Executes PyTorch tensor operations.

        Args:
            flow (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        return (torch.mean(dx) + torch.mean(dy)) / 2.0
