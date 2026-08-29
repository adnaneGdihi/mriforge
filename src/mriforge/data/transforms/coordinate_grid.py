"""Coordinate-grid emission transform (Phase 4f, Amendment I).

Attaches a normalized coordinate grid to every TorchIO Subject under
the key ``coords``. The grid is shaped (ndim, *spatial) where ``ndim``
is the spatial dimensionality of the input image (2 or 3).

Used by:

- Implicit neural representation (INR) models that take (x, y, z) ↦ value
- Coordinate-MLPs / NeRF-style architectures
- Position-aware spectral-domain models (Hyena, S4D variants)

Coordinates are in [-1, 1] per axis when ``normalize=True`` (default),
or raw integer voxel indices when False.
"""

from __future__ import annotations

import torch
import torchio as tio

from mriforge.config.schemas.data import CoordinateEmissionConfigSchema

__all__ = ["CoordinateGridTransform"]


def _make_normalized_grid(shape: tuple[int, ...], normalize: bool = True) -> torch.Tensor:
    """Build a coordinate grid of shape (ndim, *shape).

    Each output channel holds one axis of coords; ``coords[0]`` runs
    along the first spatial axis, ``coords[1]`` along the second, etc.

    Normalized coords cover [-1, 1] inclusive at the corners (matches
    F.grid_sample convention).
    """
    if normalize:
        axes = [
            torch.linspace(-1.0, 1.0, n, dtype=torch.float32)
            if n > 1
            else torch.zeros(1, dtype=torch.float32)
            for n in shape
        ]
    else:
        axes = [torch.arange(n, dtype=torch.float32) for n in shape]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=0)  # (ndim, *shape)


class CoordinateGridTransform(tio.Transform):
    """Append a normalized coordinate grid to each Subject.

    Args:
        config: Populated :class:`CoordinateEmissionConfigSchema`.

    After this transform, ``subject["coords"]`` holds the grid and
    ``subject["coord_resolution"]`` records the (W, H, D) shape used
    to build it (so downstream code can reconstruct the index mapping).
    """

    def __init__(self, config: CoordinateEmissionConfigSchema):
        super().__init__()
        self.config = config

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if not self.config.enabled:
            return subject
        input_img = subject.get("input")
        if input_img is None:
            return subject
        # input_img.data shape is (C, W, H, D); spatial = (W, H, D).
        spatial_shape = tuple(input_img.data.shape[1:])
        grid = _make_normalized_grid(spatial_shape, normalize=self.config.normalize)
        if self.config.include_batch_dim:
            grid = grid.unsqueeze(0)  # (1, ndim, *spatial)
        subject["coords"] = grid
        subject["coord_resolution"] = spatial_shape
        return subject
