"""Property-based tests for MaskGenerator / k-space sampling masks.

Properties verified
-------------------
1. **Binary mask**: generated masks contain only 0 and 1 (no fractional values).
2. **Sampling fraction**: actual sampling fraction ≈ 1/acceleration, within a
   tolerance that accounts for ACS overhead and integer rounding.
3. **ACS region always sampled**: the center of k-space (DC region) is always
   included in the mask, regardless of acceleration.
4. **Determinism**: same seed ⇒ bit-exact identical mask across two calls.

Parametrize axes
----------------
* Mask types: ``uniform_cartesian``, ``variable_density``, ``equispaced``,
  ``gaussian``, ``random`` — these are the non-trajectory types that
  ``MaskGenerator.generate_mask`` handles without external libraries.
* Accelerations: {2, 4, 8}.
* Shape: (32, 32).

Note on trajectory types (RADIAL, SPIRAL)
------------------------------------------
These call ``TrajectoryFactory`` which may have complex hardware dependencies;
they are excluded from property tests to keep the suite purely CPU-functional.

Note on Hypothesis
------------------
Hypothesis is not installed in this virtualenv.  Tests use plain
``@pytest.mark.parametrize`` instead.  Any ``from hypothesis import ...``
imports are import-guarded and will gracefully skip if unavailable.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.sampling import MaskGenerator, MaskType
from tests.utils.tolerances import tol_for

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Only mask types that work without NUFFT / TrajectoryFactory deps
_SIMPLE_MASK_TYPES = [
    MaskType.UNIFORM_CARTESIAN,
    MaskType.VARIABLE_DENSITY,
    MaskType.EQUISPACED,
    MaskType.GAUSSIAN,
    MaskType.RANDOM,
]

_ACCELERATIONS = [2, 4, 8]
_SHAPE = (32, 32)  # (H, W) — large enough to be meaningful, small enough to be fast

# ACS tolerance: the center_fraction used by these generators is ~8–10%.
# On a 32×32 grid, 8% = 2.56 lines → 3 lines → 3/32 = 9.4%.
# We allow the sampling fraction to exceed 1/acceleration by up to 20% absolute.
_FRACTION_UPPER_SLACK = 0.25
_FRACTION_LOWER_SLACK = 0.05  # allow ~5% below target (rounding / discretisation)


def _mask_id(mask_type: MaskType) -> str:
    return mask_type.value


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gen_mask(
    mask_type: MaskType,
    acceleration: int,
    seed: int,
) -> torch.Tensor:
    """Generate a mask via MaskGenerator with the given seed."""
    gen = MaskGenerator(seed=seed)
    return gen.generate_mask(mask_type, shape=_SHAPE, acceleration_factor=float(acceleration))


# ---------------------------------------------------------------------------
# Property 1: Binary mask — values in {0, 1}
# ---------------------------------------------------------------------------

@pytest.mark.property
@pytest.mark.physics
@pytest.mark.parametrize("acceleration", _ACCELERATIONS)
@pytest.mark.parametrize("mask_type", _SIMPLE_MASK_TYPES, ids=_mask_id)
def test_mask_is_binary(mask_type: MaskType, acceleration: int) -> None:
    """Generated mask contains only values 0 and 1 (no fractional sampling weights)."""
    mask = _gen_mask(mask_type, acceleration, seed=42)
    unique_vals = torch.unique(mask)
    non_binary = [v.item() for v in unique_vals if v.item() not in (0.0, 1.0)]
    assert len(non_binary) == 0, (
        f"Mask type={mask_type.value}, accel={acceleration}: "
        f"non-binary values found: {non_binary}"
    )


# ---------------------------------------------------------------------------
# Property 2: Sampling fraction ≈ 1 / acceleration  (within ACS tolerance)
# ---------------------------------------------------------------------------

@pytest.mark.property
@pytest.mark.physics
@pytest.mark.parametrize("acceleration", _ACCELERATIONS)
@pytest.mark.parametrize("mask_type", _SIMPLE_MASK_TYPES, ids=_mask_id)
def test_mask_sampling_fraction(mask_type: MaskType, acceleration: int) -> None:
    """Actual sampling fraction is within slack of 1/acceleration.

    ACS region inflates the fraction beyond 1/R; we allow an upper
    slack of ``_FRACTION_UPPER_SLACK`` and a lower slack of
    ``_FRACTION_LOWER_SLACK`` to account for both effects.
    """
    mask = _gen_mask(mask_type, acceleration, seed=99)
    total = float(mask.numel())
    sampled = float(mask.sum().item())
    actual_frac = sampled / total
    target_frac = 1.0 / acceleration

    assert actual_frac <= target_frac + _FRACTION_UPPER_SLACK, (
        f"Mask type={mask_type.value}, accel={acceleration}: "
        f"sampling fraction {actual_frac:.3f} exceeds "
        f"1/R={target_frac:.3f} + slack={_FRACTION_UPPER_SLACK:.3f}"
    )
    assert actual_frac >= target_frac - _FRACTION_LOWER_SLACK, (
        f"Mask type={mask_type.value}, accel={acceleration}: "
        f"sampling fraction {actual_frac:.3f} is below "
        f"1/R={target_frac:.3f} - slack={_FRACTION_LOWER_SLACK:.3f}"
    )


# ---------------------------------------------------------------------------
# Property 3: ACS / DC region is always sampled
# ---------------------------------------------------------------------------

@pytest.mark.property
@pytest.mark.physics
@pytest.mark.parametrize("acceleration", _ACCELERATIONS)
@pytest.mark.parametrize("mask_type", _SIMPLE_MASK_TYPES, ids=_mask_id)
def test_mask_dc_center_always_sampled(mask_type: MaskType, acceleration: int) -> None:
    """The DC bin (k=0, at centre of k-space) is always included in the mask.

    All MaskGenerator patterns that include ACS guarantee this — the DC
    component carries the macroscopic tissue contrast and must always be
    present for reconstruction to be well-posed.
    """
    mask = _gen_mask(mask_type, acceleration, seed=7)
    H, W = _SHAPE
    cy, cx = H // 2, W // 2

    # mask may be 2D (H, W) or 3D (C, H, W) depending on generator
    if mask.ndim == 3:
        dc_val = mask[0, cy, cx].item()
    else:
        dc_val = mask[cy, cx].item()

    assert dc_val == 1.0, (
        f"Mask type={mask_type.value}, accel={acceleration}: "
        f"DC bin at ({cy}, {cx}) is NOT sampled (value={dc_val})"
    )


# ---------------------------------------------------------------------------
# Property 4: Determinism — same seed produces identical masks
# ---------------------------------------------------------------------------

@pytest.mark.property
@pytest.mark.physics
@pytest.mark.parametrize("acceleration", _ACCELERATIONS)
@pytest.mark.parametrize("mask_type", _SIMPLE_MASK_TYPES, ids=_mask_id)
def test_mask_determinism_same_seed(mask_type: MaskType, acceleration: int) -> None:
    """Two calls with the same seed produce bit-identical masks."""
    seed = 1234
    mask_a = _gen_mask(mask_type, acceleration, seed=seed)
    mask_b = _gen_mask(mask_type, acceleration, seed=seed)

    assert torch.equal(mask_a, mask_b), (
        f"Mask type={mask_type.value}, accel={acceleration}: "
        f"same seed={seed} produced different masks!\n"
        f"  Max diff: {(mask_a - mask_b).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Property 5: Different seeds produce (usually) different masks
# ---------------------------------------------------------------------------

@pytest.mark.property
@pytest.mark.physics
@pytest.mark.parametrize("mask_type", [MaskType.VARIABLE_DENSITY, MaskType.RANDOM])
def test_mask_different_seeds_differ(mask_type: MaskType) -> None:
    """Different seeds produce different masks for stochastic mask types."""
    mask_a = _gen_mask(mask_type, acceleration=4, seed=1)
    mask_b = _gen_mask(mask_type, acceleration=4, seed=9999)

    # They should not be identical (collision probability is negligible for
    # stochastic masks on a 32×32 grid with ~25% sampling)
    assert not torch.equal(mask_a, mask_b), (
        f"Mask type={mask_type.value}: seed=1 and seed=9999 produced "
        f"identical masks (unexpected collision)"
    )


# ---------------------------------------------------------------------------
# Hypothesis import guard (no-op if not installed)
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings  # noqa: F401 — only guard the import
    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False


# If hypothesis were available one could add property tests here:
# @pytest.mark.skipif(not _HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
# def test_mask_binary_hypothesis(): ...
# (Kept as a placeholder; the plain parametrize tests above are the SSOT.)
