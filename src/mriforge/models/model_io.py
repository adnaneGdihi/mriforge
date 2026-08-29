"""Model input/output contract utilities for adapter auto-wiring."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelIOContract:
    """Describe the tensor interface expected or produced by a model stage."""

    rank: int
    channels: int | None = None
    spatial_dims: tuple[int, ...] | None = None
    description: str = ""

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        description: str = "",
    ) -> ModelIOContract:
        """Create a contract from a sample tensor."""
        rank = tensor.dim()
        channels = tensor.shape[1] if rank >= 2 else None
        spatial_dims: tuple[int, ...] | None
        if rank >= 3:
            spatial_dims = tuple(int(dim) for dim in tensor.shape[2:])
        else:
            spatial_dims = None
        return cls(
            rank=rank,
            channels=int(channels) if channels is not None else None,
            spatial_dims=spatial_dims,
            description=description,
        )

    def spatial_volume(self) -> int | None:
        """Return the volume implied by spatial dimensions, if available."""
        if self.spatial_dims is None:
            return None
        volume = 1
        for dim in self.spatial_dims:
            volume *= int(dim)
        return volume
