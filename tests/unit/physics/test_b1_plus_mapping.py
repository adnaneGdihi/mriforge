"""Unit tests for src/mriforge/infrastructure/physics/b1_plus_mapping.py.

Covers:
  - DoubleAngleB1Plus:  two-flip-angle B1+ estimator
  - BlochSiegertB1Plus: linearised Bloch-Siegert estimator
  - apply_b1_correction: inverse flip-angle correction helper

Physics property tests:
  - Constant-B1 region → flat map (DoubleAngleB1Plus, BlochSiegertB1Plus)
  - apply_b1_correction recovers identity on a uniform B1+ map

Differentiability:
  - DoubleAngleB1Plus fp64 gradcheck (marked @pytest.mark.gradcheck @pytest.mark.slow)
  - BlochSiegertB1Plus fp64 gradcheck (same marks)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id
from tests.utils.tolerances import tol_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mag_image(B: int, H: int, W: int, value: float = 0.5, *, seed: int | None = None) -> torch.Tensor:
    """Return a [B, 1, H, W] float32 magnitude image.

    If *value* is set, returns a constant tensor. If *seed* is set, returns
    a random tensor (ignoring *value*).
    """
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return torch.rand(B, 1, H, W, generator=g).clamp_min(0.01)
    return torch.full((B, 1, H, W), value)


def _uniform_ratio_inputs(
    B: int,
    H: int,
    W: int,
    nominal_deg: float = 60.0,
    actual_deg: float = 60.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (s_alpha, s_2alpha) consistent with a uniform actual flip angle.

    For a spoiled GRE at small flip angle, signal ~ sin(alpha), so:
        s_alpha   = sin(actual_alpha)
        s_2alpha  = sin(2 * actual_alpha)
    Returns shapes [B, 1, H, W].
    """
    alpha = math.radians(actual_deg)
    s_alpha = torch.full((B, 1, H, W), float(math.sin(alpha)))
    s_2alpha = torch.full((B, 1, H, W), float(math.sin(2 * alpha)))
    return s_alpha, s_2alpha


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_b1_plus_imports() -> None:
    """Module imports succeed without error."""
    from mriforge.infrastructure.physics.b1_plus_mapping import (
        DoubleAngleB1Plus,
        BlochSiegertB1Plus,
        apply_b1_correction,
    )
    assert DoubleAngleB1Plus is not None
    assert BlochSiegertB1Plus is not None
    assert apply_b1_correction is not None


@pytest.mark.physics
def test_canary_double_angle_tiny() -> None:
    """DoubleAngleB1Plus forward pass on 1x1x8x8 input returns correct shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    model = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    s_alpha = _mag_image(1, 8, 8)
    s_2alpha = _mag_image(1, 8, 8, value=0.8)
    out = model(s_alpha, s_2alpha)

    assert out.shape == (1, 1, 8, 8), f"Expected (1,1,8,8), got {out.shape}"
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.physics
def test_canary_bloch_siegert_tiny() -> None:
    """BlochSiegertB1Plus forward pass on 1x1x8x8 input returns correct shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    model = BlochSiegertB1Plus(kbs=74.0)
    phase_diff = torch.full((1, 1, 8, 8), math.pi / 4)
    out = model(phase_diff)

    assert out.shape == (1, 1, 8, 8)
    assert out.dtype == torch.float32
    assert (out > 0).all(), "B1+ amplitude must be positive"


@pytest.mark.physics
def test_canary_apply_b1_correction_tiny() -> None:
    """apply_b1_correction runs on minimal inputs and produces correct shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import apply_b1_correction

    image = _mag_image(1, 8, 8, value=1.0)
    b1_map = torch.ones(1, 1, 8, 8)
    out = apply_b1_correction(image, b1_map)
    assert out.shape == image.shape
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Parametrised: realistic shapes
# ---------------------------------------------------------------------------

_B1_SHAPES: list[tuple[int, int, int, int]] = [
    (1, 1, 16, 16),
    (2, 1, 32, 32),
    (1, 1, 64, 64),
    (4, 1, 32, 32),
]


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _B1_SHAPES,
    ids=[shape_id(s) for s in _B1_SHAPES],
)
def test_double_angle_output_shape(shape: tuple[int, int, int, int]) -> None:
    """DoubleAngleB1Plus output matches input shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    B, C, H, W = shape
    model = DoubleAngleB1Plus(nominal_flip_deg=45.0)
    s_alpha = torch.rand(B, C, H, W).clamp_min(0.01)
    s_2alpha = torch.rand(B, C, H, W).clamp_min(0.01)
    out = model(s_alpha, s_2alpha)
    assert out.shape == (B, C, H, W)


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _B1_SHAPES,
    ids=[shape_id(s) for s in _B1_SHAPES],
)
def test_bloch_siegert_output_shape(shape: tuple[int, int, int, int]) -> None:
    """BlochSiegertB1Plus output matches input shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    B, C, H, W = shape
    model = BlochSiegertB1Plus(kbs=74.0)
    phase_diff = torch.randn(B, C, H, W)
    out = model(phase_diff)
    assert out.shape == (B, C, H, W)


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape",
    _B1_SHAPES,
    ids=[shape_id(s) for s in _B1_SHAPES],
)
def test_apply_b1_correction_shape(shape: tuple[int, int, int, int]) -> None:
    """apply_b1_correction output matches image shape."""
    from mriforge.infrastructure.physics.b1_plus_mapping import apply_b1_correction

    B, C, H, W = shape
    image = torch.rand(B, C, H, W).clamp_min(0.01)
    b1_map = torch.rand(B, 1, H, W).clamp_min(0.1)
    out = apply_b1_correction(image, b1_map)
    assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Sanity-shape (from shape_matrices)
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "shape",
    SHAPE_MATRIX_2D,
    ids=[shape_id(s) for s in SHAPE_MATRIX_2D],
)
def test_sanity_shape_double_angle(shape: tuple[int, int, int, int]) -> None:
    """DoubleAngleB1Plus output shape matches SHAPE_MATRIX_2D contract."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    B, C, H, W = shape
    model = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    s_alpha = torch.rand(B, C, H, W).clamp_min(0.01)
    s_2alpha = torch.rand(B, C, H, W).clamp_min(0.01)
    out = model(s_alpha, s_2alpha)
    assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_double_angle_edge_b1_equals_1() -> None:
    """DoubleAngleB1Plus returns ~1.0 for the correct flip angle (s_2alpha = sin(2α))."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    nominal_deg = 60.0
    model = DoubleAngleB1Plus(nominal_flip_deg=nominal_deg)
    # Build exact double-angle pair: B1+ = 1.0 everywhere
    # ratio = cos(α) → s_2alpha / (2 * s_alpha) = cos(α)
    alpha_rad = math.radians(nominal_deg)
    s_alpha = torch.ones(1, 1, 16, 16) * math.sin(alpha_rad)
    s_2alpha = torch.ones(1, 1, 16, 16) * math.sin(2 * alpha_rad)
    out = model(s_alpha, s_2alpha)
    tol = tol_for(torch.float32) * 1000  # arccos + FP accumulation
    assert torch.allclose(out, torch.ones_like(out), atol=tol), (
        f"Expected B1+=1.0 for correct flip angle, got mean={out.mean():.5f}"
    )


@pytest.mark.physics
def test_double_angle_edge_zero_signal() -> None:
    """DoubleAngleB1Plus does not crash when s_alpha is near zero (eps guards div-by-zero)."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    model = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    s_alpha = torch.full((1, 1, 16, 16), 0.0)
    s_2alpha = torch.full((1, 1, 16, 16), 0.0)
    out = model(s_alpha, s_2alpha)
    assert not torch.isnan(out).any(), "NaN from near-zero s_alpha"


@pytest.mark.physics
def test_bloch_siegert_edge_zero_phase() -> None:
    """BlochSiegertB1Plus with zero phase returns a small positive value (eps floor)."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    model = BlochSiegertB1Plus(kbs=74.0, eps=1e-9)
    phase_diff = torch.zeros(1, 1, 16, 16)
    out = model(phase_diff)
    # eps floor: sqrt(eps) ≈ 3.16e-5
    assert (out > 0).all(), "Expected strictly positive output from eps floor"
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_bloch_siegert_edge_uniform_phase() -> None:
    """BlochSiegertB1Plus: uniform phase_diff → flat B1+ map."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    model = BlochSiegertB1Plus(kbs=74.0)
    phase_val = math.pi / 3
    phase_diff = torch.full((2, 1, 32, 32), phase_val)
    out = model(phase_diff)

    # All pixels should have the same value
    assert out.std().item() < tol_for(torch.float32), (
        f"Uniform phase did not produce flat B1+ map: std={out.std().item():.3e}"
    )


@pytest.mark.physics
def test_apply_b1_correction_edge_unit_map() -> None:
    """apply_b1_correction with b1_map=1.0 is approximately the identity."""
    from mriforge.infrastructure.physics.b1_plus_mapping import apply_b1_correction

    image = torch.rand(2, 1, 32, 32).clamp_min(0.01)
    b1_map = torch.ones(2, 1, 32, 32)
    out = apply_b1_correction(image, b1_map)
    # With eps=1e-6, out ≈ image / (1 + 1e-6) ≈ image
    tol = 1e-4
    assert torch.allclose(out, image, atol=tol), (
        f"Unit B1+ map should return ~identity, max diff={( out - image).abs().max():.3e}"
    )


@pytest.mark.physics
def test_apply_b1_correction_complex_image() -> None:
    """apply_b1_correction accepts complex images by taking .abs() internally."""
    from mriforge.infrastructure.physics.b1_plus_mapping import apply_b1_correction

    image_c = torch.complex(
        torch.rand(1, 1, 16, 16),
        torch.rand(1, 1, 16, 16),
    )
    b1_map = torch.ones(1, 1, 16, 16)
    out = apply_b1_correction(image_c, b1_map)
    assert not torch.is_complex(out), "Output should be real-valued after .abs()"
    assert out.shape == image_c.shape


# ---------------------------------------------------------------------------
# raises: documented preconditions
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_double_angle_raises_shape_mismatch() -> None:
    """DoubleAngleB1Plus raises ValueError when s_alpha and s_2alpha shapes differ."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    model = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    s_alpha = torch.rand(1, 1, 16, 16)
    s_2alpha = torch.rand(1, 1, 32, 32)  # mismatched spatial dims
    with pytest.raises(ValueError, match="shape mismatch"):
        model(s_alpha, s_2alpha)


@pytest.mark.physics
def test_bloch_siegert_raises_non_positive_kbs() -> None:
    """BlochSiegertB1Plus raises ValueError when kbs <= 0."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    with pytest.raises(ValueError, match="kbs must be > 0"):
        BlochSiegertB1Plus(kbs=0.0)

    with pytest.raises(ValueError, match="kbs must be > 0"):
        BlochSiegertB1Plus(kbs=-1.0)


@pytest.mark.physics
def test_apply_b1_correction_raises_spatial_mismatch() -> None:
    """apply_b1_correction raises ValueError when b1_map spatial dims differ."""
    from mriforge.infrastructure.physics.b1_plus_mapping import apply_b1_correction

    image = torch.rand(1, 1, 32, 32)
    b1_map = torch.rand(1, 1, 16, 16)  # wrong spatial size
    with pytest.raises(ValueError, match="spatial shape mismatch"):
        apply_b1_correction(image, b1_map)


# ---------------------------------------------------------------------------
# Physics property: constant-B1 region → flat map
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_double_angle_flat_b1_plus_map() -> None:
    """Spatially-uniform double-angle inputs produce a spatially uniform B1+ map."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    nominal_deg = 60.0
    actual_deg = 45.0
    model = DoubleAngleB1Plus(nominal_flip_deg=nominal_deg)

    alpha = math.radians(actual_deg)
    s_alpha = torch.full((1, 1, 32, 32), math.sin(alpha))
    s_2alpha = torch.full((1, 1, 32, 32), math.sin(2 * alpha))

    out = model(s_alpha, s_2alpha)
    # All pixels must have the same B1+ estimate (flat map)
    assert out.std().item() < tol_for(torch.float32) * 100, (
        f"Constant B1 region did not yield flat map: std={out.std().item():.3e}"
    )
    # The absolute B1+ value must be close to actual/nominal
    expected = actual_deg / nominal_deg
    assert abs(out.mean().item() - expected) < 0.01, (
        f"B1+ mean {out.mean():.4f} far from expected {expected:.4f}"
    )


@pytest.mark.physics
def test_bloch_siegert_flat_b1_plus_map() -> None:
    """Spatially-uniform phase_diff produces a spatially uniform B1+ map."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    kbs = 74.0
    phase_val = 0.5  # 0.5 rad uniform phase shift
    model = BlochSiegertB1Plus(kbs=kbs)

    phase_diff = torch.full((1, 1, 32, 32), phase_val)
    out = model(phase_diff)

    # std << mean → flat map
    rel_std = out.std().item() / (out.mean().item() + 1e-12)
    assert rel_std < 1e-4, (
        f"Uniform phase_diff did not give flat B1+ map: rel_std={rel_std:.3e}"
    )


# ---------------------------------------------------------------------------
# Differentiability / gradcheck (slow, opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.gradcheck
@pytest.mark.slow
def test_gradcheck_double_angle() -> None:
    """DoubleAngleB1Plus is differentiable: fp64 gradcheck passes."""
    from mriforge.infrastructure.physics.b1_plus_mapping import DoubleAngleB1Plus

    model = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    # Use small spatial size to keep gradcheck fast
    # fp64 inputs required by torch.autograd.gradcheck
    alpha = math.radians(60.0)
    s_a = torch.full((1, 1, 4, 4), math.sin(alpha), dtype=torch.float64, requires_grad=True)
    s_2a = torch.full((1, 1, 4, 4), math.sin(2 * alpha), dtype=torch.float64, requires_grad=True)

    # Cast model to float64
    model_64 = model.double()

    assert torch.autograd.gradcheck(
        model_64,
        (s_a, s_2a),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    ), "gradcheck failed for DoubleAngleB1Plus"


@pytest.mark.physics
@pytest.mark.gradcheck
@pytest.mark.slow
def test_gradcheck_bloch_siegert() -> None:
    """BlochSiegertB1Plus is differentiable: fp64 gradcheck passes."""
    from mriforge.infrastructure.physics.b1_plus_mapping import BlochSiegertB1Plus

    model = BlochSiegertB1Plus(kbs=74.0)
    model_64 = model.double()

    phase_diff = torch.full(
        (1, 1, 4, 4), 0.4, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        model_64,
        (phase_diff,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        raise_exception=True,
    ), "gradcheck failed for BlochSiegertB1Plus"
