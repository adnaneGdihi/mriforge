"""Tests for the Connes spectral-triple intertwining loss + Dirac operator."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.spectral_triple import (
    connes_distance_upper_bound,
    discrete_dirac_2d,
)
from spectramr.models.losses.spectral_triple_loss import SpectralTripleIntertwiningLoss


def test_dirac_2d_shape() -> None:
    """The Dirac operator preserves shape."""
    psi = torch.randn(2, 2, 8, 8)
    out = discrete_dirac_2d(psi, field_strength=1.0)
    assert out.shape == psi.shape


def test_dirac_zero_on_constant_spinor() -> None:
    """Constant fields are in the kernel of the gradient → Dirac action vanishes."""
    psi = torch.ones(2, 2, 8, 8)
    out = discrete_dirac_2d(psi, field_strength=1.0)
    # Interior gradient is zero on a constant; boundary uses one-sided diffs
    # which also vanish on a constant.
    assert torch.allclose(out.abs(), torch.zeros_like(out.abs()), atol=1e-6)


def test_dirac_scales_linearly_with_field() -> None:
    """D_{B0} is linear in B0 by construction."""
    psi = torch.randn(2, 8, 8)
    out_1 = discrete_dirac_2d(psi, field_strength=1.0)
    out_2 = discrete_dirac_2d(psi, field_strength=2.0)
    assert torch.allclose(out_2, 2.0 * out_1, atol=1e-5)


def test_dirac_invalid_shape_raises() -> None:
    bad = torch.randn(3, 4, 4)  # not 2 spinor channels
    with pytest.raises(ValueError):
        discrete_dirac_2d(bad)


def test_connes_distance_upper_bound() -> None:
    """Cauchy-Schwarz: d ≤ ||a-b||_2."""
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    d = connes_distance_upper_bound(a, b)
    assert d.item() == pytest.approx((a - b).reshape(-1).norm().item(), rel=1e-6)


def test_intertwining_loss_zero_on_identity() -> None:
    """When prediction == target == ulf_state, both terms vanish (field_ratio = 1)."""
    psi = torch.randn(1, 2, 8, 8)
    loss = SpectralTripleIntertwiningLoss(b0_ulf=1.0, b0_hf=1.0)
    out = loss(psi, psi, ulf_state=psi)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_intertwining_loss_positive_on_perturbation() -> None:
    psi = torch.randn(1, 2, 8, 8)
    perturb = psi + 0.1 * torch.randn_like(psi)
    loss = SpectralTripleIntertwiningLoss()
    out = loss(perturb, psi, ulf_state=psi)
    assert out.item() > 0.0


def test_intertwining_loss_handles_single_channel_input() -> None:
    """Single-channel images are zero-padded to 2-component spinors."""
    psi = torch.randn(1, 1, 8, 8)
    loss = SpectralTripleIntertwiningLoss()
    out = loss(psi, psi, ulf_state=psi)
    assert torch.isfinite(out)


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        SpectralTripleIntertwiningLoss(intertwining_weight=-1.0)
    with pytest.raises(ValueError):
        SpectralTripleIntertwiningLoss(b0_ulf=0.0)
    with pytest.raises(ValueError):
        SpectralTripleIntertwiningLoss(reduction="not_a_reduction")
