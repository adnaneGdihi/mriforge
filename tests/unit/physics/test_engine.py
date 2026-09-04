"""Regression tests for ``PhysicsEngine`` knob-wiring contracts.

Targets ``spectramr.infrastructure.physics.engine``. These tests pin the
post-fix contracts from the WS-4 forward-operator review:

* ``dc_type`` dispatch is exhaustive: an advertised-but-unwired / unknown value
  (e.g. ``physics_informed``) RAISES instead of silently degrading to ``soft``
  (pitfall #9).
* ``enforce_data_consistency`` does not silently drop ``sensitivity_maps``:
  modules whose ``forward`` cannot accept smaps raise ``NotImplementedError``;
  the adaptive (SENSE) module forwards them (pitfall #15).
* ``image_to_kspace`` / ``kspace_to_image`` reject ``centered=False`` rather
  than silently ignoring the knob (pitfall #15 + physics SSOT).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.engine import PhysicsEngine


# ---------------------------------------------------------------------------
# dc_type dispatch (no silent fallback to soft)
# ---------------------------------------------------------------------------


def test_unknown_dc_type_raises() -> None:
    """An unwired/unknown dc_type must raise, never silently degrade to soft."""
    engine = PhysicsEngine(dc_type="physics_informed")  # advertised-but-unwired
    with pytest.raises(ValueError, match="Unknown dc_type"):
        engine.get_dc_module()


def test_soft_dc_type_still_builds_simple_module() -> None:
    from spectramr.infrastructure.physics.data_consistency import SimpleDataConsistency

    engine = PhysicsEngine(dc_type="soft")
    assert isinstance(engine.get_dc_module(), SimpleDataConsistency)


# ---------------------------------------------------------------------------
# enforce_data_consistency sensitivity_maps wiring
# ---------------------------------------------------------------------------


def test_enforce_dc_rejects_smaps_on_single_coil_module() -> None:
    """Passing sensitivity_maps to a soft DC engine must raise, not silently
    drop the coil maps (pitfall #15)."""
    engine = PhysicsEngine(dc_type="soft")
    img = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    smaps = torch.randn(1, 4, 16, 16, dtype=torch.complex64)
    with pytest.raises(NotImplementedError):
        engine.enforce_data_consistency(img, k, mask, sensitivity_maps=smaps)


def test_enforce_dc_forwards_smaps_on_adaptive_module() -> None:
    """The adaptive DC module accepts smaps and runs SENSE-based DC."""
    engine = PhysicsEngine(dc_type="adaptive")
    img = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    smaps = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    out = engine.enforce_data_consistency(img, k, mask, sensitivity_maps=smaps)
    assert out.shape == img.shape


def test_enforce_dc_single_coil_still_works_without_smaps() -> None:
    """The no-smaps path is unchanged (numerics-neutral regression guard)."""
    engine = PhysicsEngine(dc_type="soft")
    img = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    out = engine.enforce_data_consistency(img, k, mask)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# centered knob (reject the inert value)
# ---------------------------------------------------------------------------


def test_image_to_kspace_rejects_uncentered() -> None:
    engine = PhysicsEngine()
    img = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="centered"):
        engine.image_to_kspace(img, centered=False)


def test_kspace_to_image_rejects_uncentered() -> None:
    engine = PhysicsEngine()
    k = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="centered"):
        engine.kspace_to_image(k, centered=False)


def test_centered_true_round_trip_unchanged() -> None:
    """centered=True (default) still does a clean fft/ifft round-trip."""
    engine = PhysicsEngine()
    img = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    k = engine.image_to_kspace(img, centered=True)
    recon = engine.kspace_to_image(k, centered=True)
    assert torch.allclose(recon, img, atol=1e-5)
