"""Phase 4f Amendment I — CoordinateGridTransform + schema."""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from mriforge.config.schemas.data import (
    CoordinateEmissionConfigSchema,
    DataConfigSchema,
)
from mriforge.data.transforms.coordinate_grid import (
    CoordinateGridTransform,
    _make_normalized_grid,
)


# ── Schema ──────────────────────────────────────────────────────────────────


def test_coordinate_emission_defaults() -> None:
    cfg = CoordinateEmissionConfigSchema()
    assert cfg.enabled is False
    assert cfg.normalize is True
    assert cfg.include_batch_dim is True


def test_data_config_threads_emit_coordinates_block() -> None:
    cfg = DataConfigSchema(
        emit_coordinates={
            "enabled": True,
            "normalize": False,
            "include_batch_dim": False,
        }
    )
    assert cfg.emit_coordinates.enabled is True
    assert cfg.emit_coordinates.normalize is False


# ── Grid construction ──────────────────────────────────────────────────────


def test_normalized_grid_3d_has_correct_shape() -> None:
    g = _make_normalized_grid((4, 4, 2), normalize=True)
    assert g.shape == (3, 4, 4, 2)


def test_normalized_grid_2d_has_correct_shape() -> None:
    g = _make_normalized_grid((8, 8), normalize=True)
    assert g.shape == (2, 8, 8)


def test_normalized_grid_range_is_pm1() -> None:
    g = _make_normalized_grid((4, 4, 4), normalize=True)
    assert g.min() == pytest.approx(-1.0)
    assert g.max() == pytest.approx(1.0)


def test_unnormalized_grid_holds_integer_indices() -> None:
    g = _make_normalized_grid((3, 4, 2), normalize=False)
    # Last axis values should run [0, 1] for size 2.
    assert torch.allclose(g[2, 0, 0, :], torch.tensor([0.0, 1.0]))


def test_single_voxel_axis_collapses_to_zero() -> None:
    """Avoid divide-by-zero: shape with a 1-voxel axis ⇒ that axis is all 0."""
    g = _make_normalized_grid((4, 4, 1), normalize=True)
    assert g.shape == (3, 4, 4, 1)
    assert torch.all(g[2] == 0.0)


# ── Transform on a TorchIO Subject ─────────────────────────────────────────


def test_transform_disabled_is_noop() -> None:
    """When config.enabled=False, the Subject must be returned unchanged."""
    cfg = CoordinateEmissionConfigSchema()  # disabled
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    assert "coords" not in out


def test_transform_appends_coords_and_resolution() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True)
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    assert "coords" in out
    assert "coord_resolution" in out
    assert out["coord_resolution"] == (4, 4, 2)


def test_transform_include_batch_dim_adds_leading_dim() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, include_batch_dim=True)
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    # Shape: (1, ndim, *spatial) = (1, 3, 4, 4, 2)
    assert out["coords"].shape == (1, 3, 4, 4, 2)


def test_transform_without_batch_dim() -> None:
    cfg = CoordinateEmissionConfigSchema(enabled=True, include_batch_dim=False)
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    assert out["coords"].shape == (3, 4, 4, 2)


def test_transform_handles_missing_input_gracefully() -> None:
    """No 'input' key ⇒ no-op (don't crash)."""
    cfg = CoordinateEmissionConfigSchema(enabled=True)
    subj = tio.Subject(target=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = CoordinateGridTransform(cfg)(subj)
    # No 'coords' added since there was no 'input' to size against.
    assert "coords" not in out
