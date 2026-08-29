"""Tests for B1+ (transmit-field) mapping primitives.

Targets ``mriforge.infrastructure.physics.b1_plus_mapping``. Verifies the
double-angle and Bloch-Siegert estimators recover known B1+ values on
synthetic inputs, and that shape mismatches fail loud.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.b1_plus_mapping import (
    BlochSiegertB1Plus,
    DoubleAngleB1Plus,
    apply_b1_correction,
)


# ---------------------------------------------------------------------------
# DoubleAngleB1Plus
# ---------------------------------------------------------------------------


def test_double_angle_recovers_nominal_flip_when_b1_is_unity() -> None:
    """If actual flip == nominal, |S2| / 2|S1| = cos(α) → recovered B1 = 1.

    For the small-flip approximation, S(α) ≈ sin(α) and S(2α) ≈ sin(2α) =
    2·sin(α)·cos(α), so |S2| / (2|S1|) = cos(α). arccos(cos(α)) = α.
    Relative B1+ = α / α_nominal = 1.
    """
    nominal_deg = 60.0
    estimator = DoubleAngleB1Plus(nominal_flip_deg=nominal_deg)
    alpha = math.radians(nominal_deg)
    s_alpha = torch.full((1, 1, 4, 4), math.sin(alpha))
    s_2alpha = torch.full((1, 1, 4, 4), math.sin(2 * alpha))
    out = estimator(s_alpha, s_2alpha)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4), (
        f"got {out[0, 0, 0, 0].item():.4f}, expected 1.0"
    )


def test_double_angle_recovers_half_b1() -> None:
    """If actual flip = nominal / 2, the recovered ratio = 0.5."""
    nominal_deg = 60.0
    estimator = DoubleAngleB1Plus(nominal_flip_deg=nominal_deg)
    actual = math.radians(nominal_deg / 2)
    s_alpha = torch.full((1, 1, 4, 4), math.sin(actual))
    s_2alpha = torch.full((1, 1, 4, 4), math.sin(2 * actual))
    out = estimator(s_alpha, s_2alpha)
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=2e-3)


def test_double_angle_shape_mismatch_raises() -> None:
    """Mismatched shapes must raise (CLAUDE.md #9)."""
    estimator = DoubleAngleB1Plus()
    with pytest.raises(ValueError, match="shape mismatch"):
        estimator(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))


def test_double_angle_complex_input_uses_magnitude() -> None:
    """Complex-valued s_alpha / s_2alpha are reduced to magnitude internally."""
    estimator = DoubleAngleB1Plus(nominal_flip_deg=60.0)
    alpha = math.radians(60.0)
    real = torch.full((1, 1, 4, 4), math.sin(alpha))
    cplx = torch.complex(real, torch.zeros_like(real))
    real2 = torch.full((1, 1, 4, 4), math.sin(2 * alpha))
    cplx2 = torch.complex(real2, torch.zeros_like(real2))
    out_real = estimator(real, real2)
    out_cplx = estimator(cplx, cplx2)
    assert torch.allclose(out_real, out_cplx, atol=1e-5)


# ---------------------------------------------------------------------------
# BlochSiegertB1Plus
# ---------------------------------------------------------------------------


def test_bloch_siegert_zero_phase_yields_minimal_b1() -> None:
    """Zero phase → near-zero B1+ (sqrt(eps))."""
    estimator = BlochSiegertB1Plus(kbs=74.0)
    out = estimator(torch.zeros(1, 1, 4, 4))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_bloch_siegert_recovers_b1_from_known_phase() -> None:
    """B1 = sqrt(|phase| / kbs) for the linearised model."""
    kbs = 74.0
    estimator = BlochSiegertB1Plus(kbs=kbs)
    phase = torch.full((1, 1, 4, 4), 5.0)  # rad
    out = estimator(phase)
    expected = math.sqrt(5.0 / kbs)
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-3)


def test_bloch_siegert_invalid_kbs_raises() -> None:
    """kbs ≤ 0 fails loud."""
    with pytest.raises(ValueError, match="kbs must be > 0"):
        BlochSiegertB1Plus(kbs=0.0)


# ---------------------------------------------------------------------------
# apply_b1_correction
# ---------------------------------------------------------------------------


def test_apply_b1_correction_identity_when_b1_is_unity() -> None:
    """B1 = 1 → image unchanged (within numerical tolerance)."""
    image = torch.rand(1, 1, 8, 8) + 0.1  # avoid zeros
    b1 = torch.ones(1, 1, 8, 8)
    out = apply_b1_correction(image, b1)
    assert torch.allclose(out, image, atol=1e-3)


def test_apply_b1_correction_scales_inversely() -> None:
    """B1 = 0.5 → image divided by 0.5 → output = 2 × input."""
    image = torch.full((1, 1, 4, 4), 1.0)
    b1 = torch.full((1, 1, 4, 4), 0.5)
    out = apply_b1_correction(image, b1)
    assert torch.allclose(out, torch.full_like(out, 2.0), atol=1e-3)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32), (4, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_double_angle_shape_preservation(shape: tuple[int, int, int, int]) -> None:
    estimator = DoubleAngleB1Plus()
    s_alpha = torch.rand(*shape) + 0.5
    s_2alpha = torch.rand(*shape) + 0.5
    out = estimator(s_alpha, s_2alpha)
    assert out.shape == shape
    assert torch.isfinite(out).all()
