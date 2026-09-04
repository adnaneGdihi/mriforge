"""Tests for coordinate-sampling utilities (INR helpers).

Targets ``spectramr.infrastructure.physics.coordinate_utils``. The
``CoordinateSampler`` provides regular-grid, random-uniform, and
ROI-restricted coordinate sampling for INR training.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.coordinate_utils import CoordinateSampler


# ---------------------------------------------------------------------------
# get_full_grid
# ---------------------------------------------------------------------------


def test_full_grid_shape_is_b_h_w_2() -> None:
    """Output is ``[B, H*W, 2]``."""
    grid = CoordinateSampler.get_full_grid(2, (4, 8), torch.device("cpu"))
    assert grid.shape == (2, 32, 2)


def test_full_grid_corners_are_minus_one_to_one() -> None:
    """Grid spans ``[-1, 1]`` in both dimensions exactly at the corners."""
    grid = CoordinateSampler.get_full_grid(1, (4, 4), torch.device("cpu"))
    flat = grid[0]  # [H*W, 2]
    # Min / max should be -1 / +1 in each dim.
    assert torch.allclose(flat[..., 0].min(), torch.tensor(-1.0))
    assert torch.allclose(flat[..., 0].max(), torch.tensor(1.0))
    assert torch.allclose(flat[..., 1].min(), torch.tensor(-1.0))
    assert torch.allclose(flat[..., 1].max(), torch.tensor(1.0))


def test_full_grid_is_deterministic() -> None:
    """Two calls with the same args produce identical grids."""
    g1 = CoordinateSampler.get_full_grid(1, (4, 4), torch.device("cpu"))
    g2 = CoordinateSampler.get_full_grid(1, (4, 4), torch.device("cpu"))
    assert torch.equal(g1, g2)


def test_full_grid_batch_replication() -> None:
    """Each batch entry holds an identical grid."""
    grid = CoordinateSampler.get_full_grid(3, (2, 2), torch.device("cpu"))
    assert torch.equal(grid[0], grid[1])
    assert torch.equal(grid[1], grid[2])


# ---------------------------------------------------------------------------
# sample_random_points
# ---------------------------------------------------------------------------


def test_random_points_shape() -> None:
    """``sample_random_points`` returns ``[B, n, 2]``."""
    pts = CoordinateSampler.sample_random_points(2, 100, torch.device("cpu"))
    assert pts.shape == (2, 100, 2)


def test_random_points_within_minus_one_to_one() -> None:
    """All sampled coordinates lie in ``[-1, 1]``."""
    pts = CoordinateSampler.sample_random_points(4, 200, torch.device("cpu"))
    assert (pts >= -1.0).all()
    assert (pts <= 1.0).all()


def test_random_points_change_between_calls() -> None:
    """Random sampling actually randomises (not a fixed grid)."""
    p1 = CoordinateSampler.sample_random_points(1, 100, torch.device("cpu"))
    p2 = CoordinateSampler.sample_random_points(1, 100, torch.device("cpu"))
    assert not torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# sample_roi
# ---------------------------------------------------------------------------


def test_roi_clamped_to_minus_one_to_one() -> None:
    """ROI samples are clamped to ``[-1, 1]`` even when ROI extends beyond."""
    pts = CoordinateSampler.sample_roi(
        batch_size=2, n_points=200, center=(0.9, 0.9),
        scale=1.0, device=torch.device("cpu"),
    )
    assert (pts >= -1.0).all()
    assert (pts <= 1.0).all()


def test_roi_centred_at_origin_with_small_scale() -> None:
    """A small ROI at origin produces points near origin."""
    pts = CoordinateSampler.sample_roi(
        batch_size=1, n_points=500, center=(0.0, 0.0),
        scale=0.1, device=torch.device("cpu"),
    )
    # All points within radius 0.2 (= 2 * scale due to (rand - 0.5) * 2 * scale).
    assert pts.abs().max().item() <= 0.2 + 1e-5


def test_roi_shape() -> None:
    """ROI sampling shape: ``[B, n_points, 2]``."""
    pts = CoordinateSampler.sample_roi(
        3, 50, center=(0.0, 0.0), scale=0.5, device=torch.device("cpu"),
    )
    assert pts.shape == (3, 50, 2)


@pytest.mark.parametrize(
    "h,w",
    [(4, 4), (8, 8), (16, 32), (1, 1)],
    ids=lambda v: f"{v}",
)
def test_full_grid_arbitrary_resolution(h: int, w: int) -> None:
    """Grid generation handles arbitrary spatial dims, including 1×1."""
    grid = CoordinateSampler.get_full_grid(1, (h, w), torch.device("cpu"))
    assert grid.shape == (1, h * w, 2)
    assert torch.isfinite(grid).all()
