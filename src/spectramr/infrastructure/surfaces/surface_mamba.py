"""Cortex-adaptive bidirectional Mamba scan block.

Operates on the disk-flattened representation produced by
:class:`ConformalFlattening`. Composes:

1. A cortex-adaptive SFC permutation (derived from a
   lesion-likelihood or retinotopy prior — defaults to raster).
2. A bidirectional 1-D scan (Conv1d stand-in for the full Mamba
   kernel; the framework already ships a real Mamba block for
   production use).

Used by ``cortical_conformal_fmri_recon`` (fMRI §3) when a real
surface flattening is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurfaceMambaBlock(nn.Module):
    """Bidirectional 1-D scan over a cortex-adaptive SFC.

    Args:
        channels: Feature channels.
        kernel_size: Receptive-field width of the scan.
    """

    def __init__(self, channels: int, *, kernel_size: int = 7) -> None:
        super().__init__()
        self.fwd = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.bwd = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.proj = nn.Conv1d(channels * 2, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, *, sfc_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            x: ``[B, C, H, W]`` disk-flattened cortex feature tensor.
            sfc_indices: Optional ``[B, H·W]`` permutation. Raster
                scan when absent.
        """
        if x.dim() != 4:
            raise ValueError(f"SurfaceMambaBlock expects 4-D input, got {x.shape}")
        B, C, H, W = x.shape
        flat = x.view(B, C, H * W)
        if sfc_indices is not None:
            flat = flat.gather(2, sfc_indices.unsqueeze(1).expand(B, C, -1))
        f = self.fwd(flat)
        b = self.bwd(flat.flip(-1)).flip(-1)
        merged = self.act(self.proj(torch.cat([f, b], dim=1)))
        if sfc_indices is not None:
            out = flat.new_zeros(B, C, H * W)
            out.scatter_(2, sfc_indices.unsqueeze(1).expand(B, C, -1), merged)
            merged = out
        return x + merged.view(B, C, H, W)


__all__ = ["SurfaceMambaBlock"]
