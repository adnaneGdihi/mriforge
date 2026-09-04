"""Unit tests for coordinate_grid transform.

Covers:
  - _make_normalized_grid: shape, range ([-1,1]), unnormalised integer indices,
    single-voxel axis collapses to zero (no divide-by-zero).
  - CoordinateGridTransform:
    - canary on a Subject,
    - disabled config is no-op,
    - subject gains 'coords' and 'coord_resolution',
    - include_batch_dim=True adds leading 1,
    - include_batch_dim=False omits it,
    - missing 'input' key is handled gracefully (no crash).
  - CoordinateEmissionConfigSchema: defaults and field acceptance.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.config.schemas.data import CoordinateEmissionConfigSchema
from spectramr.data.transforms.coordinate_grid import (
    CoordinateGridTransform,
    _make_normalized_grid,
)


# ── _make_normalized_grid ─────────────────────────────────────────────────────


def test_canary_normalized_grid_3d_shape() -> None:
    g = _make_normalized_grid((4, 4, 2), normalize=True)
    assert g.shape == (3, 4, 4, 2)


def test_normalized_grid_2d_shape() -> None:
    g = _make_normalized_grid((8, 8), normalize=True)
    assert g.shape == (2, 8, 8)


def test_normalized_grid_range_is_pm1() -> None:
    g = _make_normalized_grid((4, 4, 4), normalize=True)
    assert g.min().item() == pytest.approx(-1.0)
    assert g.max().item() == pytest.approx(1.0)


def test_unnormalized_grid_holds_integer_indices() -> None:
    """Last axis (size=2) should contain [0.0, 1.0]."""
    g = _make_normalized_grid((3, 4, 2), normalize=False)
    expected = torch.tensor([0.0, 1.0])
    assert torch.allclose(g[2, 0, 0, :], expected)


def test_single_voxel_axis_collapses_to_zero() -> None:
    """Avoid divide-by-zero: a 1-voxel axis is mapped to 0.0."""
    g = _make_normalized_grid((4, 4, 1), normalize=True)
    assert g.shape == (3, 4, 4, 1)
    assert torch.all(g[2] == 0.0)


# ── CoordinateEmissionConfigSchema ────────────────────────────────────────────


def test_schema_defaults() -> None:
    cfg = CoordinateEmissionConfigSchema()
    assert cfg.enabled is False
    assert cfg.normalize is True
    assert cfg.include_batch_dim is True


def test_schema_enabled_flag_accepted() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, normalize=False, include_batch_dim=False)
    assert cfg.enabled is True
    assert cfg.normalize is False
    assert cfg.include_batch_dim is False


# ── CoordinateGridTransform ───────────────────────────────────────────────────


def _make_subject(shape: tuple[int, ...] = (1, 4, 4, 2)) -> tio.Subject:
    return tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(*shape)))


def test_canary_transform_appends_coords() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True)
    out = CoordinateGridTransform(cfg)(_make_subject())
    assert "coords" in out


def test_disabled_transform_is_noop() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=False)
    out = CoordinateGridTransform(cfg)(_make_subject())
    assert "coords" not in out


def test_transform_appends_coord_resolution() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True)
    out = CoordinateGridTransform(cfg)(_make_subject())
    assert "coord_resolution" in out
    assert out["coord_resolution"] == (4, 4, 2)


def test_transform_include_batch_dim_true() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, include_batch_dim=True)
    out = CoordinateGridTransform(cfg)(_make_subject())
    # (1, ndim=3, 4, 4, 2)
    assert out["coords"].shape == (1, 3, 4, 4, 2)


def test_transform_include_batch_dim_false() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, include_batch_dim=False)
    out = CoordinateGridTransform(cfg)(_make_subject())
    assert out["coords"].shape == (3, 4, 4, 2)


def test_transform_coords_range_is_pm1() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, normalize=True, include_batch_dim=False)
    out = CoordinateGridTransform(cfg)(_make_subject((1, 8, 8, 4)))
    coords = out["coords"]
    assert coords.min().item() == pytest.approx(-1.0)
    assert coords.max().item() == pytest.approx(1.0)


def test_transform_unnormalized_coords_are_integer_indices() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, normalize=False, include_batch_dim=False)
    out = CoordinateGridTransform(cfg)(_make_subject((1, 3, 4, 2)))
    coords = out["coords"]
    # All values should be non-negative integers
    assert (coords >= 0).all()
    assert torch.allclose(coords, coords.floor())


def test_transform_missing_input_key_no_crash() -> None:
    """Subject without 'input' → no coords added, no exception."""
    cfg = CoordinateEmissionConfigSchema(enabled=True)
    subj = tio.Subject(target=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    assert "coords" not in out
