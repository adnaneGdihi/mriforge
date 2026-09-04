"""Soft Dynamic Time Warping Loss for misregistered MRI pairs.

Computes a differentiable approximation of Dynamic Time Warping between
predicted and target 1D sequences.  Designed for use with Hilbert-linearized
MRI sequences where paired ULF/HF scans may not be perfectly registered.

A spatial shift in 3D mathematically results in localized insertions/deletions
in the 1D Hilbert sequence.  Standard L1/L2 losses produce blurry averages
under such misalignment.  Soft-DTW allows elastic alignment while maintaining
differentiability.

To keep computation tractable, the loss operates on **windowed subsequences**
of length ``window_size`` (default 64), giving O(B · K · w²) complexity where
K = L / stride_size.

Math:
    DTW(x, y) = min_π Σ_{(i,j)∈π} d(x_i, y_j)
    Soft-DTW replaces min with soft-min: min^γ(a) = -γ log Σ exp(-a/γ)

Reference:
    Cuturi & Blondel (2017). "Soft-DTW: a Differentiable Loss Function for
    Time-Series." ICML 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


def _soft_dtw_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute Soft-DTW distance between two sequences.

    Args:
        pred: Predicted sequence ``[B, L, D]``.
        target: Target sequence ``[B, L, D]``.
        gamma: Soft-min smoothing parameter.

    Returns:
        Scalar DTW distance per batch element ``[B]``.
    """
    B, N, D = pred.shape
    _, M, _ = target.shape

    # Pairwise squared Euclidean distance: [B, N, M]
    dist = torch.cdist(pred, target, p=2.0).pow(2)

    # DP table: [B, N+1, M+1] initialized to +inf
    R = torch.full((B, N + 1, M + 1), float("inf"), device=pred.device, dtype=pred.dtype)
    R[:, 0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # soft-min over three predecessors
            r0 = R[:, i - 1, j - 1]
            r1 = R[:, i - 1, j]
            r2 = R[:, i, j - 1]

            stack = torch.stack([r0, r1, r2], dim=-1)  # [B, 3]
            soft_min = -gamma * torch.logsumexp(-stack / gamma, dim=-1)

            R[:, i, j] = dist[:, i - 1, j - 1] + soft_min

    return R[:, N, M]


@register_loss("soft_dtw", aliases=["SoftDTW", "sdtw"])
class SoftDTWLoss(nn.Module):
    """Windowed Soft-DTW loss for Hilbert-linearized MRI sequences.

    Applies Soft-DTW on non-overlapping windows of the flattened sequence
    to keep memory and compute tractable.

    Args:
        gamma: Soft-min smoothing parameter (lower → harder alignment).
        window_size: Size of each DTW window.
        stride_size: Stride between windows (default = window_size).
        normalize: Whether to normalize by window count.

    Example::

        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        loss = loss_fn(prediction, target)  # prediction/target: [B, C, H, W]
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{SoftDTW} = \text{SoftDTW}_\\gamma(\\hat{y}, y)"""

    def __init__(
        self,
        gamma: float = 0.1,
        window_size: int = 64,
        stride_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.window_size = window_size
        self.stride_size = stride_size or window_size
        self.normalize = normalize

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute windowed Soft-DTW loss.

        Args:
            prediction: Predicted tensor ``[B, C, H, W]`` or ``[B, L, D]``.
            target: Target tensor (same shape as prediction).

        Returns:
            Scalar loss value.

        forward method for SoftDTWLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Volumetric (5-D) inputs: the Soft-DTW sequence is per-slice, so
        # fold the trailing depth axis into the batch dimension (TorchIO
        # emits [B, C, H, W, D]) and align each in-plane slice independently.
        # For D == 1 this is a plain squeeze; for D > 1 each slice becomes its
        # own DTW alignment. This is what lets volumetric Hilbert-Mamba arms
        # (e.g. exp_hm_08_sdtw_mamba, hdsf_mamba_unet backbone) use the loss
        # at all — the pre-2026-06 contract raised on any 5-D input.
        if prediction.dim() == 5:
            b, c, h, w, depth = prediction.shape
            prediction = prediction.permute(0, 4, 1, 2, 3).reshape(b * depth, c, h, w)
            target = target.permute(0, 4, 1, 2, 3).reshape(b * depth, c, h, w)

        # Flatten to [B, L, D] if spatial
        if prediction.dim() == 4:
            B, C, H, W = prediction.shape
            pred_seq = prediction.reshape(B, C, -1).permute(0, 2, 1)
            tgt_seq = target.reshape(B, C, -1).permute(0, 2, 1)
        elif prediction.dim() == 3:
            pred_seq = prediction
            tgt_seq = target
        else:
            raise ValueError(f"SoftDTWLoss expects 3D, 4D, or 5D input, got {prediction.dim()}D")

        B, L, D = pred_seq.shape
        total_loss = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
        num_windows = 0

        # Windowed DTW
        for start in range(0, L - self.window_size + 1, self.stride_size):
            end = start + self.window_size
            pred_win = pred_seq[:, start:end, :]
            tgt_win = tgt_seq[:, start:end, :]
            dtw_dist = _soft_dtw_distance(pred_win, tgt_win, self.gamma)
            total_loss = total_loss + dtw_dist.mean()
            num_windows += 1

        if num_windows == 0:
            # Sequence shorter than window: compute on full sequence
            dtw_dist = _soft_dtw_distance(pred_seq, tgt_seq, self.gamma)
            return dtw_dist.mean()

        if self.normalize:
            total_loss = total_loss / num_windows

        return total_loss


__all__ = ["SoftDTWLoss"]
