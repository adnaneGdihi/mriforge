"""Tests for ``DifferentiableBlochSimulator`` (SE / GRE).

Targets ``mriforge.infrastructure.physics.bloch_simulation``. This is the
simulator consumed by the MRF dictionary; it implements the SE and GRE
analytical signal equations as a differentiable PyTorch module.

A separate suite covers ``DifferentiableBlochLayer`` (SPGR-only); this
file focuses on the SE / GRE-capable simulator.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.bloch_simulation import (
    DifferentiableBlochSimulator,
)


def _tissue(b: int, h: int, w: int, pd: float, t1: float, t2: float) -> torch.Tensor:
    """Build a [B, 3, H, W] tissue map with constant (PD, T1, T2)."""
    pd_map = torch.full((b, 1, h, w), pd)
    t1_map = torch.full((b, 1, h, w), t1)
    t2_map = torch.full((b, 1, h, w), t2)
    return torch.cat([pd_map, t1_map, t2_map], dim=1)


# ---------------------------------------------------------------------------
# SE
# ---------------------------------------------------------------------------


def test_se_matches_analytical_formula() -> None:
    """SE: S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2)."""
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    tr, te = 500.0, 30.0
    out = sim(tissue, TR=tr, TE=te)
    expected = (1.0 - math.exp(-tr / 1000.0)) * math.exp(-te / 80.0)
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)


def test_se_zero_pd_yields_zero() -> None:
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = _tissue(1, 4, 4, pd=0.0, t1=1000.0, t2=80.0)
    out = sim(tissue, TR=500.0, TE=30.0)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_gre_flip_angle_uses_full_precision_pi() -> None:
    """GRE at alpha=180 deg must yield sin(alpha)~=0 to the float32 precision floor.

    Regression for flag F-D: the degrees->radians conversion previously used the
    6-digit literal ``3.14159`` instead of ``torch.pi``, leaving a ~2.7e-6 residual
    in ``sin(180 deg)`` that leaked into the GRE signal as ~1.4e-7. With
    full-precision pi the residual collapses to the float32 floor (~4.8e-9, set by
    float32's rounding of pi, ~8.7e-8 in ``sin``). The 1e-8 threshold separates the
    buggy literal (fails) from the fixed conversion (passes).
    """
    sim = DifferentiableBlochSimulator(sequence_type="GRE")
    tissue = _tissue(1, 2, 2, pd=1.0, t1=900.0, t2=80.0)
    sig = sim(tissue, TR=100.0, TE=1.0, alpha=180.0)
    assert sig.abs().max().item() < 1e-8


def test_se_short_te_preserves_more_signal() -> None:
    """Shorter TE → less T2 decay → larger signal."""
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    out_short = sim(tissue, TR=500.0, TE=10.0)
    out_long = sim(tissue, TR=500.0, TE=200.0)
    assert out_short.mean().item() > out_long.mean().item()


# ---------------------------------------------------------------------------
# GRE (Ernst equation)
# ---------------------------------------------------------------------------


def test_gre_matches_ernst_equation() -> None:
    """GRE: S = PD * sin(α) * (1-E1) / (1 - cos(α)·E1) * E2."""
    sim = DifferentiableBlochSimulator(sequence_type="GRE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    tr, te, alpha_deg = 10.0, 3.0, 15.0
    out = sim(tissue, TR=tr, TE=te, alpha=alpha_deg)
    e1 = math.exp(-tr / 1000.0)
    e2 = math.exp(-te / 80.0)
    alpha_rad = alpha_deg * 3.14159 / 180.0
    expected = (
        math.sin(alpha_rad) * (1 - e1) / (1 - math.cos(alpha_rad) * e1) * e2
    )
    # Tolerance loose because the simulator uses 3.14159 not math.pi.
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-4)


def test_gre_zero_flip_yields_zero() -> None:
    """sin(0) = 0 → GRE signal vanishes."""
    sim = DifferentiableBlochSimulator(sequence_type="GRE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    out = sim(tissue, TR=10.0, TE=3.0, alpha=0.0)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_gre_default_alpha_is_90() -> None:
    """When ``alpha`` is None, the simulator uses 90° as default."""
    sim = DifferentiableBlochSimulator(sequence_type="GRE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    out_none = sim(tissue, TR=10.0, TE=3.0, alpha=None)
    out_90 = sim(tissue, TR=10.0, TE=3.0, alpha=90.0)
    assert torch.allclose(out_none, out_90, atol=1e-5)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_sequence_raises() -> None:
    """Sequence types other than SE/GRE fail loud (CLAUDE.md #9)."""
    sim = DifferentiableBlochSimulator(sequence_type="FSE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0)
    with pytest.raises(ValueError, match="Unknown sequence type"):
        sim(tissue, TR=10.0, TE=3.0)


def test_signal_is_non_negative_after_abs() -> None:
    """The simulator applies ``abs()`` to the tissue map → signal ≥ 0."""
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    # Even with a deliberately negative PD, abs() makes the signal positive.
    pd = torch.full((1, 1, 4, 4), -0.5)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    tissue = torch.cat([pd, t1, t2], dim=1)
    out = sim(tissue, TR=500.0, TE=30.0)
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 8, 8), (1, 32, 32), (2, 16, 16), (4, 8, 8)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_se_shape_preservation(shape: tuple[int, int, int]) -> None:
    """SE output preserves [B, 1, H, W] across the parametrised matrix."""
    b, h, w = shape
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = _tissue(b, h, w, pd=1.0, t1=800.0, t2=60.0)
    out = sim(tissue, TR=500.0, TE=30.0)
    assert out.shape == (b, 1, h, w)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_through_tissue_maps() -> None:
    """Backward through SE / GRE reaches the tissue parameters."""
    sim = DifferentiableBlochSimulator(sequence_type="SE")
    tissue = _tissue(1, 4, 4, pd=1.0, t1=1000.0, t2=80.0).requires_grad_(True)
    out = sim(tissue, TR=500.0, TE=30.0)
    out.sum().backward()
    assert tissue.grad is not None
    assert torch.isfinite(tissue.grad).all()
    assert tissue.grad.abs().max().item() > 0
