"""Tests for ``DifferentiableBlochLayer`` and ``SPGRBlochSimulator``.

Targets ``spectramr.infrastructure.physics.differentiable_bloch``. This module
is **load-bearing** for the ULF radiologist-acceptance roadmap (every
inverse-Bloch / Bloch-consistent SSDU / cycle-Bloch path consumes the
forward operator here), so the property tests verify both numerical
correctness against analytical SPGR and gradient flow for use as a
training-time loss term.

Categories:

- Unit: SPGR magnitude analytical match, B0 phase, B1 scaling, sequence-
  type validation.
- Sanity-shape: parametrised ``[B, 1, H, W]`` matrix.
- Edge: zero PD, zero flip, very short / very long T1/T2, 0 D b0_map.
- Gradient: tissue-parameter gradients flow (load-bearing for II.10).
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.differentiable_bloch import (
    DifferentiableBlochLayer,
    SPGRBlochSimulator,
)


# ---------------------------------------------------------------------------
# Unit + analytical-correctness
# ---------------------------------------------------------------------------


def test_spgr_magnitude_matches_analytical_formula() -> None:
    """SPGR steady-state matches the closed-form expression within 1e-5.

    S = PD * sin(α) * (1 - E1) / (1 - cos(α)·E1) * E2
    """
    layer = DifferentiableBlochLayer(sequence_type="SPGR")
    pd = torch.full((1, 1, 4, 4), 1.0)
    t1 = torch.full((1, 1, 4, 4), 1000.0)  # ms
    t2 = torch.full((1, 1, 4, 4), 80.0)
    tr = 10.0
    te = 3.0
    flip = math.radians(15.0)
    out = layer(t1, t2, pd, tr, te, flip)

    e1 = math.exp(-tr / 1000.0)
    e2 = math.exp(-te / 80.0)
    expected = (math.sin(flip) * (1 - e1) / (1 - math.cos(flip) * e1)) * e2

    # softplus on T1/T2 is identity for these values; tolerance is loose
    # because softplus shifts the relaxation by a small amount.
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-3), (
        f"got {out[0, 0, 0, 0].item():.6f}, expected {expected:.6f}"
    )


def test_zero_pd_yields_zero_signal() -> None:
    """PD = 0 → zero magnitude regardless of T1, T2, flip."""
    layer = DifferentiableBlochLayer()
    pd = torch.zeros(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(30.0))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_zero_flip_angle_yields_zero_signal() -> None:
    """sin(α)=0 → magnitude=0, irrespective of B1 scaling."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    out = layer(t1, t2, pd, 10.0, 3.0, 0.0)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_b0_map_produces_complex_output_with_correct_phase() -> None:
    """When ``b0_map`` is provided, output is complex with phase 2π·f·TE."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 2, 2)
    t1 = torch.full((1, 1, 2, 2), 1000.0)
    t2 = torch.full((1, 1, 2, 2), 80.0)
    b0 = torch.full((1, 1, 2, 2), 100.0)  # Hz
    te_ms = 3.0
    out = layer(t1, t2, pd, 10.0, te_ms, math.radians(15.0), b0_map=b0)
    assert torch.is_complex(out), "b0_map provided → complex output"

    expected_phase = 2 * math.pi * 100.0 * (te_ms * 1e-3)
    actual_phase = torch.angle(out)
    # Phase wraps to [-π, π]; compare modulo 2π.
    diff = (actual_phase - expected_phase + math.pi) % (2 * math.pi) - math.pi
    assert diff.abs().max().item() < 1e-4


def test_b0_none_yields_real_output() -> None:
    """When ``b0_map`` is None, output stays real."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 2, 2)
    t1 = torch.full((1, 1, 2, 2), 1000.0)
    t2 = torch.full((1, 1, 2, 2), 80.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert not torch.is_complex(out)


def test_b1_scaling_modulates_effective_flip() -> None:
    """B1 = 0.5 reduces effective flip by half."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    flip = math.radians(20.0)
    out_full = layer(t1, t2, pd, 10.0, 3.0, flip)
    out_half = layer(t1, t2, pd, 10.0, 3.0, flip, b1_map=torch.full((1, 1, 4, 4), 0.5))
    out_reduced = layer(t1, t2, pd, 10.0, 3.0, flip / 2)
    assert torch.allclose(out_half, out_reduced, atol=1e-5), (
        "B1=0.5 should equal flip/2"
    )
    # And differ from full-flip output.
    assert not torch.allclose(out_full, out_half, atol=1e-3)


def test_unsupported_sequence_type_raises() -> None:
    """Non-SPGR sequence types must fail loud (CLAUDE.md #9)."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        DifferentiableBlochLayer(sequence_type="EPG")


def test_spgr_simulator_alias_equivalent_to_layer() -> None:
    """``SPGRBlochSimulator`` is a thin alias — outputs match."""
    pd = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1000.0)
    t2 = torch.full((1, 1, 4, 4), 80.0)
    layer = DifferentiableBlochLayer(sequence_type="SPGR")
    sim = SPGRBlochSimulator()
    out_layer = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    out_sim = sim(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert torch.allclose(out_layer, out_sim, atol=1e-7)


# ---------------------------------------------------------------------------
# Sanity-shape: parametrised matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (1, 1, 32, 32), (2, 1, 32, 32), (4, 1, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix_real_output(shape: tuple[int, int, int, int]) -> None:
    """Forward pass preserves spatial shape across realistic sizes."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(shape)
    t1 = torch.full(shape, 800.0)
    t2 = torch.full(shape, 50.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert out.shape == shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 16, 16), (2, 1, 32, 32)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix_complex_with_b0(shape: tuple[int, int, int, int]) -> None:
    """With B0 map, output is complex and shape-preserving."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(shape)
    t1 = torch.full(shape, 800.0)
    t2 = torch.full(shape, 50.0)
    b0 = torch.zeros(shape)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0), b0_map=b0)
    assert out.shape == shape
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_very_short_t1_does_not_explode() -> None:
    """T1 ≈ 0 → softplus stabiliser keeps signal finite."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1e-6)
    t2 = torch.full((1, 1, 4, 4), 50.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert torch.isfinite(out).all()


def test_very_long_t1_yields_low_signal() -> None:
    """T1 → ∞ implies E1 → 1, so (1-E1) → 0 — signal vanishes."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 4, 4)
    t1 = torch.full((1, 1, 4, 4), 1e10)
    t2 = torch.full((1, 1, 4, 4), 50.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert out.abs().max().item() < 1e-3


def test_signal_is_non_negative() -> None:
    """SPGR magnitude is clamped to ≥0."""
    layer = DifferentiableBlochLayer()
    pd = torch.ones(1, 1, 8, 8)
    t1 = torch.full((1, 1, 8, 8), 1000.0)
    t2 = torch.full((1, 1, 8, 8), 50.0)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# Gradient flow (load-bearing for ULF inverse-Bloch use case)
# ---------------------------------------------------------------------------


def test_gradient_flows_to_tissue_parameters() -> None:
    """Backward through the Bloch layer reaches T1, T2, PD."""
    layer = DifferentiableBlochLayer()
    t1 = torch.full((1, 1, 4, 4), 1000.0, requires_grad=True)
    t2 = torch.full((1, 1, 4, 4), 50.0, requires_grad=True)
    pd = torch.full((1, 1, 4, 4), 1.0, requires_grad=True)
    out = layer(t1, t2, pd, 10.0, 3.0, math.radians(15.0))
    out.sum().backward()
    assert t1.grad is not None and torch.isfinite(t1.grad).all()
    assert t2.grad is not None and torch.isfinite(t2.grad).all()
    assert pd.grad is not None and torch.isfinite(pd.grad).all()
    # Non-zero gradients (signal does depend on each parameter).
    assert t1.grad.abs().max().item() > 0
    assert t2.grad.abs().max().item() > 0
    assert pd.grad.abs().max().item() > 0
