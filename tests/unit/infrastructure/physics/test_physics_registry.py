"""Tests for the physics-parameter registry.

Targets ``spectramr.infrastructure.physics.physics_registry``. Provides
canonical (TR, TE, TI, B0) parameters for standard MRI sequences,
plus name-based inference and tensor (de)normalisation helpers.

Categories:

- ``PhysicsParameters.to_normalized_tensor`` matches the documented
  scaling
- ``to_tensor`` returns raw (un-normalised) values
- Round-trip: ``denormalize(normalize(x)) == x``
- ``PHYSICS_REGISTRY`` covers every documented sequence
- ``infer_physics_from_name`` dispatches ULF / FLAIR / DWI / T1w / T2w
  / PD / HF / fallback correctly
- ULF entries always have ``B0 = 0.064``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.physics_registry import (
    PHYSICS_REGISTRY,
    PhysicsParameters,
    denormalize_physics_tensor,
    infer_physics_from_name,
    normalize_physics_tensor,
)


# ---------------------------------------------------------------------------
# PhysicsParameters
# ---------------------------------------------------------------------------


def test_to_normalized_tensor_shape() -> None:
    """``to_normalized_tensor`` returns a 4-vector."""
    p = PHYSICS_REGISTRY["T1w"]
    out = p.to_normalized_tensor()
    assert out.shape == (4,)


def test_to_normalized_tensor_values_in_unit_interval() -> None:
    """Each component lies in ``[0, 1]`` after normalisation."""
    p = PHYSICS_REGISTRY["FLAIR"]
    out = p.to_normalized_tensor()
    assert (out >= 0).all()
    assert (out <= 1.0 + 1e-6).all()


def test_to_tensor_returns_raw_values() -> None:
    """``to_tensor`` preserves the raw (TR, TE, TI, B0) tuple."""
    p = PHYSICS_REGISTRY["T2w"]
    out = p.to_tensor()
    assert out[0].item() == p.TR
    assert out[1].item() == p.TE
    assert out[2].item() == p.TI
    assert out[3].item() == p.B0


# ---------------------------------------------------------------------------
# Round-trip normalize/denormalize
# ---------------------------------------------------------------------------


def test_normalize_denormalize_round_trip() -> None:
    """``denormalize(normalize(x)) ≈ x``."""
    raw = torch.tensor([2000.0, 80.0, 0.0, 3.0])
    norm = normalize_physics_tensor(raw)
    recovered = denormalize_physics_tensor(norm)
    assert torch.allclose(recovered, raw)


def test_normalize_does_not_mutate_input() -> None:
    """Normalisation clones — input unchanged."""
    raw = torch.tensor([2000.0, 80.0, 0.0, 3.0])
    raw_copy = raw.clone()
    _ = normalize_physics_tensor(raw)
    assert torch.equal(raw, raw_copy)


def test_normalize_batched_input() -> None:
    """Batched ``[N, 4]`` inputs work the same way."""
    raw = torch.tensor([[2000.0, 80.0, 0.0, 3.0], [300.0, 20.0, 0.0, 0.064]])
    norm = normalize_physics_tensor(raw)
    assert norm.shape == raw.shape
    # B0 of the second row: 0.064 / 7 ≈ 0.00914
    assert pytest.approx(norm[1, 3].item(), abs=1e-4) == 0.064 / 7.0


# ---------------------------------------------------------------------------
# Coverage of registry entries
# ---------------------------------------------------------------------------


def test_canonical_entries_present() -> None:
    """All documented contrast keys are in the registry."""
    for key in (
        "T1w", "T2w", "FLAIR", "DWI", "PD",
        "ULF_T1w", "ULF_T2w", "ULF",
        "HF", "T1", "T2",
    ):
        assert key in PHYSICS_REGISTRY, f"missing {key!r}"


def test_ulf_entries_have_low_field_b0() -> None:
    """All ULF aliases use 64 mT."""
    for key in ("ULF", "ULF_T1w", "ULF_T2w", "ULF_64mT"):
        assert PHYSICS_REGISTRY[key].B0 == 0.064


def test_high_field_entries_use_3t() -> None:
    """Standard HF entries are at 3 T."""
    for key in ("T1w", "T2w", "FLAIR", "DWI", "PD", "HF"):
        assert PHYSICS_REGISTRY[key].B0 == 3.0


# ---------------------------------------------------------------------------
# Name-based inference
# ---------------------------------------------------------------------------


def test_infer_t1w() -> None:
    """``T1w`` maps to the canonical T1w entry."""
    p = infer_physics_from_name("T1w")
    assert p.contrast_type == "T1w"


def test_infer_flair() -> None:
    """``FLAIR`` is recognised."""
    p = infer_physics_from_name("FLAIR")
    assert p.contrast_type == "FLAIR"
    assert p.TI == 2500.0


def test_infer_dwi_via_diffusion() -> None:
    """``"diffusion"`` keyword routes to DWI."""
    p = infer_physics_from_name("brain_diffusion_b1000")
    assert p.contrast_type == "DWI"


def test_infer_ulf_t2w_from_filename() -> None:
    """A ULF + T2w filename routes to ``ULF_T2w``."""
    p = infer_physics_from_name("ulf_registered_sub01_T2w.nii.gz")
    assert p.contrast_type == "ULF_T2w"


def test_infer_ulf_t1w_from_filename() -> None:
    """A ULF + T1w filename routes to ``ULF_T1w``."""
    p = infer_physics_from_name("ULF_T1w_brain.nii.gz")
    assert p.contrast_type == "ULF_T1w"


def test_infer_generic_ulf() -> None:
    """``"ULF"`` alone routes to the generic ULF entry."""
    p = infer_physics_from_name("ulf_phantom.nii.gz")
    assert p.contrast_type == "ULF"


def test_infer_unknown_falls_back_to_t1w() -> None:
    """Completely unknown name → T1w fallback (per docstring)."""
    p = infer_physics_from_name("totally_unrecognised")
    assert p.contrast_type == "T1w"


def test_infer_pd() -> None:
    """``PD`` keyword routes to proton density."""
    p = infer_physics_from_name("PD_volume")
    assert p.contrast_type == "PD"
