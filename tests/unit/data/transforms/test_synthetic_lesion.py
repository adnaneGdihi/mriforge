"""Tests for ``SyntheticLesionTransform`` validation paths.

Targets ``spectramr.data.transforms.synthetic_lesion``. The full transform
exercises ``MultiPhysicsBlochLayer`` over volumetric tissue parameters
— that's covered in the integration tier. Here we restrict scope to
the validation surface (preset names, custom-dict completeness, range
checks) and the Gaussian-blob helper.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.transforms.synthetic_lesion import (
    _LESION_PRESETS,
    _TISSUE_TABLE_3T,
    SyntheticLesionTransform,
    _gaussian_blob_3d,
)


# ---------------------------------------------------------------------------
# Preset table & tissue table
# ---------------------------------------------------------------------------


def test_lesion_presets_have_required_keys() -> None:
    """Every preset has ``d_rho``, ``d_t1``, ``d_t2`` shifts."""
    for name, preset in _LESION_PRESETS.items():
        assert {"d_rho", "d_t1", "d_t2"} <= preset.keys(), (
            f"preset {name!r} missing required keys"
        )


def test_tissue_table_has_canonical_classes() -> None:
    """``_TISSUE_TABLE_3T`` covers WM, GM, CSF and background."""
    assert {"wm", "gm", "csf", "background"} <= _TISSUE_TABLE_3T.keys()
    for cls, params in _TISSUE_TABLE_3T.items():
        assert {"rho", "t1", "t2"} <= params.keys()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_construction_unknown_preset_raises() -> None:
    """Unknown lesion preset → ``ValueError`` listing valid presets."""
    with pytest.raises(ValueError, match="lesion_type must be one of"):
        SyntheticLesionTransform(lesion_type="not_a_real_lesion")


def test_construction_custom_dict_missing_keys_raises() -> None:
    """Custom dict missing any of ``d_rho|d_t1|d_t2`` → ``ValueError``."""
    with pytest.raises(ValueError, match="custom lesion_type must contain"):
        SyntheticLesionTransform(lesion_type={"d_rho": 0.1})  # missing d_t1, d_t2


def test_construction_custom_dict_complete_accepted() -> None:
    """Complete custom dict is accepted and stored."""
    custom = {"d_rho": 0.0, "d_t1": 100.0, "d_t2": 50.0}
    transform = SyntheticLesionTransform(lesion_type=custom)
    assert transform.shifts == custom
    assert transform.lesion_label == "custom"


def test_construction_zero_n_lesions_raises() -> None:
    """``n_lesions < 1`` → ``ValueError``."""
    with pytest.raises(ValueError, match="n_lesions must be >= 1"):
        SyntheticLesionTransform(n_lesions=0)


def test_construction_negative_radius_raises() -> None:
    """Non-positive radius → ``ValueError``."""
    with pytest.raises(ValueError, match="radius_voxels must be positive"):
        SyntheticLesionTransform(radius_voxels=(0.0, 5.0))


def test_construction_negative_jitter_raises() -> None:
    """Negative jitter → ``ValueError``."""
    with pytest.raises(ValueError, match="intensity_jitter must be >= 0"):
        SyntheticLesionTransform(intensity_jitter=-0.1)


def test_construction_persists_attributes() -> None:
    """Construction stores config attributes for downstream introspection."""
    transform = SyntheticLesionTransform(
        lesion_type="wmh",
        n_lesions=3,
        radius_voxels=(2.0, 5.0),
        intensity_jitter=0.2,
        seed=42,
    )
    assert transform.lesion_label == "wmh"
    assert transform.n_lesions == 3
    assert transform.radius_voxels == (2.0, 5.0)
    assert transform.intensity_jitter == 0.2
    assert transform.seed == 42


# ---------------------------------------------------------------------------
# Bloch sub-network is frozen
# ---------------------------------------------------------------------------


def test_bloch_layer_frozen_for_augmentation() -> None:
    """The internal ``MultiPhysicsBlochLayer`` is in eval mode and not learnable."""
    transform = SyntheticLesionTransform()
    assert transform._bloch.training is False
    for p in transform._bloch.parameters():
        assert p.requires_grad is False


# ---------------------------------------------------------------------------
# Gaussian blob helper
# ---------------------------------------------------------------------------


def test_gaussian_blob_peak_at_centre() -> None:
    """The blob attains its maximum at the centre voxel."""
    blob = _gaussian_blob_3d(
        shape=(16, 16, 16),
        centre=(8, 8, 8),
        radius=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert blob[8, 8, 8] == blob.max()


def test_gaussian_blob_decays_with_distance() -> None:
    """Voxels further from centre have lower amplitude."""
    blob = _gaussian_blob_3d(
        shape=(16, 16, 16),
        centre=(8, 8, 8),
        radius=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    near = blob[8, 9, 8]   # 1 voxel from centre
    far = blob[8, 14, 8]   # 6 voxels from centre
    assert near > far


def test_gaussian_blob_shape_matches_request() -> None:
    """Blob shape equals the requested ``(X, Y, Z)``."""
    blob = _gaussian_blob_3d(
        shape=(8, 16, 24),
        centre=(4, 8, 12),
        radius=3.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert blob.shape == (8, 16, 24)


def test_gaussian_blob_radius_zero_is_safe() -> None:
    """Zero radius is clamped to a small positive sigma → no NaN."""
    blob = _gaussian_blob_3d(
        shape=(4, 4, 4),
        centre=(2, 2, 2),
        radius=0.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.isfinite(blob).all()


# ---------------------------------------------------------------------------
# Crude tissue lookup
# ---------------------------------------------------------------------------


def test_crude_tissue_params_returns_three_maps() -> None:
    """``_crude_tissue_params`` returns ρ, T1, T2 maps with the same shape as input."""
    image = torch.rand(8, 8, 8) * 0.8 + 0.1
    out = SyntheticLesionTransform._crude_tissue_params(image, brain_mask=None)
    assert {"rho", "t1", "t2"} == out.keys()
    for v in out.values():
        assert v.shape == image.shape
        assert torch.isfinite(v).all()


def test_crude_tissue_params_uses_brain_mask_when_given() -> None:
    """When a brain mask is provided, voxels outside default to background."""
    image = torch.rand(8, 8, 8) * 0.8 + 0.1
    mask = torch.zeros(8, 8, 8)
    mask[2:6, 2:6, 2:6] = 1.0  # only central cube is brain
    out = SyntheticLesionTransform._crude_tissue_params(image, brain_mask=mask)
    # Outside the mask, ρ should equal background-ρ (= 0.0).
    assert (out["rho"][0, 0, 0]).item() == _TISSUE_TABLE_3T["background"]["rho"]
