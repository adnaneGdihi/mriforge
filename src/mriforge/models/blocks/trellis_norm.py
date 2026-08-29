"""TRELLIS normalization layers with float32 stability.

Migrated from src/implementations/trellis/modules/norm.py during consolidation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    """LayerNorm that normalises in float32 for numerical stability."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm variant mirroring the LayerNorm32 behaviour."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return super().forward(x.float()).type(x.dtype)


class ChannelLayerNorm32(LayerNorm32):
    """Channel-wise LayerNorm operating along the channel dimension."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        dim = x.dim()
        x_permuted = x.permute(0, *range(2, dim), 1).contiguous()
        normed = super().forward(x_permuted)
        return normed.permute(0, dim - 1, *range(1, dim - 1)).contiguous()


__all__ = ["ChannelLayerNorm32", "GroupNorm32", "LayerNorm32"]
