"""Task 5: PMA-VarNet Global Objective Loss.

Implements the joint multi-contrast reconstruction loss anchoring to the digital marker.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.ssim_loss import SSIMLoss

logger = logging.getLogger(__name__)


@register_loss("pma_global", aliases=["PMAGlobalLoss"])
class PMAGlobalLoss(nn.Module):
    r"""Joint Multi-Contrast Objective Loss for PMA-VarNet.

    Optimizes the network over all contrasts (T1w, T2w, FLAIR) simultaneously.
    Combines:
    1. L1 global anatomy loss
    2. Structural Similarity (SSIM)
    3. L2 Marker Pinning Prior (strictly enforcing synthetic geometry)

    .. math::

        \mathcal{L} = \sum_{c=1}^{N_{con}} \left(
            \|\hat{\mathbf{x}}_c - \mathbf{x}_{c, gt}\|_1 +
            \lambda_1 (1 - \text{SSIM}) +
            \lambda_2 \|\mathbf{M}_{syn} \odot (\hat{\mathbf{x}}_c - \mathbf{m}_{syn, c})\|_2^2
        \right)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_marker: float = 5.0,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_marker = lambda_marker
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the joint multi-contrast loss.

        Args:
            prediction: Reconstructed images `[B, N_con, H, W]`
            target: Ground truth images `[B, N_con, H, W]`
            marker_mask: Binary mask `[1, 1, H, W]` or `[B, 1, H, W]`
            marker_prior: Ideal marker `[B, N_con, H, W]` (or broadcastable)

        Returns:
            Scalar objective loss.

        forward method for PMAGlobalLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")

        B, N_con, H, W = prediction.shape
        total_loss = torch.tensor(0.0, device=prediction.device)

        # SSIM typically expects real-valued [B, 1, H, W], we take magnitude if complex
        pred_mag = prediction.abs() if prediction.is_complex() else prediction
        targ_mag = target.abs() if target.is_complex() else target
        # Resolve the SSIM data range ONCE (it is the global magnitude max,
        # constant across channels). Previously this ``.item()`` ran inside the
        # per-channel loop — one GPU→host sync per channel per step. (SSIMLoss's
        # data_range is a Python float by API, so a single sync is unavoidable;
        # hoisting it out of the loop removes the N_con-fold redundancy.)
        ssim_data_range = float(targ_mag.max())

        for c in range(N_con):
            p_c = prediction[:, c : c + 1]
            t_c = target[:, c : c + 1]
            pm_c = pred_mag[:, c : c + 1]
            tm_c = targ_mag[:, c : c + 1]

            # 1. L1 Loss (complex-safe: use magnitude for complex tensors)
            if p_c.is_complex():
                l1 = (p_c - t_c).abs().mean()
            else:
                l1 = F.l1_loss(p_c, t_c)

            # 2. SSIM Loss (1 - SSIM); SSIMLoss returns a scalar tensor.
            # No silent fallback: a previous bare ``except Exception`` replaced a
            # failed SSIM with a constant 1.0, dropping the structural term while
            # the run looked green (pitfall #10). Let genuine errors surface.
            l_ssim = self.ssim_loss(pm_c, tm_c, data_range=ssim_data_range)

            # 3. Marker Prior L2
            l_marker = torch.tensor(0.0, device=prediction.device)
            if marker_mask is not None and marker_prior is not None:
                # Ensure broadcasting across channels if needed
                prior_c = marker_prior[:, c : c + 1] if marker_prior.shape[1] > 1 else marker_prior
                p_c_masked = p_c * marker_mask
                prior_c_masked = prior_c * marker_mask
                # [FIX] mse_cuda not implemented for ComplexFloat
                # Use manual MSE via abs().pow(2) for complex tensors
                if p_c_masked.is_complex():
                    l_marker = (p_c_masked - prior_c_masked).abs().pow(2).mean()
                else:
                    l_marker = F.mse_loss(p_c_masked, prior_c_masked)

            c_loss = (
                (self.lambda_l1 * l1)
                + (self.lambda_ssim * l_ssim)
                + (self.lambda_marker * l_marker)
            )
            total_loss = total_loss + c_loss

        return total_loss


__all__ = ["PMAGlobalLoss"]
