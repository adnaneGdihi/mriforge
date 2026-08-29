"""Extended tests for sampling.py (critical-partial module).

Existing canaries for UniformCartesianKSpaceAccelerator and KSpaceAccelerator
live in test_physics_modules.py — we EXTEND coverage here WITHOUT touching that file.

New coverage:
  - MaskType enum (illegal string → must raise, per CLAUDE.md pitfall #9)
  - MaskGenerator.generate_mask() for all supported mask types
  - MaskGenerator.get_gaussian_pdf / get_variable_density_pdf
  - Edge: tiny shapes, B=1 shapes
  - Raises: unknown mask type string
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_mask_type_enum():
    """MaskType enum contains expected members and importing works."""
    from mriforge.infrastructure.physics.sampling import MaskType
    assert MaskType.UNIFORM_CARTESIAN.value == "uniform_cartesian"
    assert MaskType.CARTESIAN_LINES.value == "cartesian_lines"
    assert MaskType.VARIABLE_DENSITY.value == "variable_density"
    assert MaskType.RADIAL.value == "radial"
    assert MaskType.GAUSSIAN.value == "gaussian"
    assert MaskType.EQUISPACED.value == "equispaced"
    assert MaskType.RANDOM.value == "random"


@pytest.mark.physics
def test_canary_mask_generator_instantiates():
    """MaskGenerator instantiates with and without seed."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg_no_seed = MaskGenerator()
    mg_seeded = MaskGenerator(seed=42)
    assert mg_no_seed is not None
    assert mg_seeded is not None


# ---------------------------------------------------------------------------
# Parametrized — generate_mask for Cartesian variants
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "mask_type_str",
    ["uniform_cartesian", "cartesian_lines", "equispaced"],
)
@pytest.mark.parametrize(
    "h, w, accel",
    [
        (32, 32, 2.0),
        (64, 32, 4.0),   # rectangular
        (32, 64, 2.0),
    ],
    ids=["32x32-r2", "64x32-r4-rect", "32x64-r2"],
)
def test_generate_cartesian_mask_shape(mask_type_str, h, w, accel):
    """Cartesian mask types return [H, W] float mask."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=0)
    mask = mg.generate_mask(mask_type_str, shape=(h, w), acceleration_factor=accel)
    assert mask.shape[-2:] == (h, w), f"Expected (..., {h}, {w}) got {tuple(mask.shape)}"
    assert mask.sum() > 0, "Mask is all zeros"


@pytest.mark.physics
@pytest.mark.parametrize(
    "h, w, accel",
    [
        (32, 32, 2.0),
        (64, 64, 4.0),
    ],
    ids=["32x32-r2", "64x64-r4"],
)
def test_generate_variable_density_mask_shape(h, w, accel):
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=1)
    mask = mg.generate_mask("variable_density", shape=(h, w), acceleration_factor=accel)
    assert mask.shape[-2:] == (h, w)
    assert mask.sum() > 0


@pytest.mark.physics
@pytest.mark.parametrize(
    "h, w, accel",
    [
        (32, 32, 2.0),
        (64, 64, 4.0),
    ],
    ids=["32x32", "64x64"],
)
def test_generate_gaussian_mask_shape(h, w, accel):
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=2)
    mask = mg.generate_mask("gaussian", shape=(h, w), acceleration_factor=accel)
    assert mask.shape[-2:] == (h, w)
    assert mask.sum() > 0


@pytest.mark.physics
@pytest.mark.parametrize("h, w", [(32, 32), (64, 64), (32, 64)])
def test_generate_random_mask_shape(h, w):
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=3)
    mask = mg.generate_mask("random", shape=(h, w), acceleration_factor=2.0)
    assert mask.shape[-2:] == (h, w)


# ---------------------------------------------------------------------------
# Radial mask
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize("h, w", [(32, 32), (64, 64)])
def test_generate_radial_mask_shape(h, w):
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=4)
    mask = mg.generate_mask("radial", shape=(h, w), acceleration_factor=4.0)
    assert mask.shape[-2:] == (h, w)


# ---------------------------------------------------------------------------
# Cartesian peripheral
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_generate_cartesian_peripheral_shape():
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=5)
    mask = mg.generate_mask("cartesian_peripheral", shape=(32, 32), acceleration_factor=2.0)
    assert mask.shape[-2:] == (32, 32)


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize("num_lines", [32, 64, 128])
def test_gaussian_pdf_sums_to_one(num_lines):
    import numpy as np
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    pdf = MaskGenerator.get_gaussian_pdf(num_lines)
    assert abs(pdf.sum() - 1.0) < 1e-6
    assert (pdf >= 0).all()


@pytest.mark.physics
@pytest.mark.parametrize("num_lines", [32, 64, 128])
def test_variable_density_pdf_sums_to_one(num_lines):
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    pdf = MaskGenerator.get_variable_density_pdf(num_lines)
    assert abs(pdf.sum() - 1.0) < 1e-6
    assert (pdf >= 0).all()


# ---------------------------------------------------------------------------
# Reproducibility (seeded)
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_cartesian_mask_reproducible_with_seed():
    """Two MaskGenerator instances with same seed generate identical masks."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg1 = MaskGenerator(seed=42)
    mg2 = MaskGenerator(seed=42)
    m1 = mg1.generate_mask("cartesian_lines", shape=(32, 32), acceleration_factor=4.0)
    m2 = mg2.generate_mask("cartesian_lines", shape=(32, 32), acceleration_factor=4.0)
    assert torch.equal(m1, m2)


# ---------------------------------------------------------------------------
# Shape matrix
# ---------------------------------------------------------------------------

@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, c, h, w",
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D if h <= 64 and w <= 64],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D if h <= 64 and w <= 64],
)
def test_cartesian_mask_shape_matrix(b, c, h, w):
    """Cartesian mask [H, W] fits the spatial dims of SHAPE_MATRIX_2D entries."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=0)
    mask = mg.generate_mask("cartesian_lines", shape=(h, w), acceleration_factor=4.0)
    assert mask.shape[-2:] == (h, w)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_mask_generator_edge_high_acceleration():
    """Acceleration factor near H still produces a mask (no crash)."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=0)
    mask = mg.generate_mask("cartesian_lines", shape=(16, 16), acceleration_factor=8.0)
    assert mask.shape == (16, 16)


@pytest.mark.physics
def test_mask_generator_edge_accel_1():
    """Acceleration factor = 1 → nearly fully sampled."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator(seed=0)
    mask = mg.generate_mask("cartesian_lines", shape=(32, 32), acceleration_factor=1.0)
    # Most lines should be sampled
    frac = float(mask[:, 0].float().mean().item())
    assert frac > 0.5, f"Expected mostly-sampled mask, got frac={frac}"


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests — CLAUDE.md pitfall #9: must raise, not fall back
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_mask_type_raises_on_illegal_string():
    """MaskType(illegal_string) must raise ValueError — no silent fallback."""
    from mriforge.infrastructure.physics.sampling import MaskType
    with pytest.raises(ValueError):
        MaskType("completely_invalid_mask_type")


@pytest.mark.physics
def test_mask_generator_raises_on_unknown_mask_type():
    """generate_mask() with an unknown string must raise ValueError."""
    from mriforge.infrastructure.physics.sampling import MaskGenerator
    mg = MaskGenerator()
    with pytest.raises(ValueError):
        mg.generate_mask("not_a_real_mask", shape=(32, 32))
