"""Tests for the stochastic-resetting diffusion modifier."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.diffusion.stochastic_resetting import (
    apply_resetting,
    coefficient_of_variation,
    mfpt_with_resetting,
    optimal_resetting_rate,
)


def test_mfpt_at_zero_rate_equals_mean() -> None:
    """L'Hôpital limit: τ_r → E[τ₀] as r → 0."""
    samples = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tau = mfpt_with_resetting(samples, 0.0)
    assert tau.item() == pytest.approx(3.0, abs=1e-6)


def test_mfpt_at_exponential_is_constant() -> None:
    """For exponential τ₀ ~ Exp(λ=1), CoV=1 exactly and τ_r is r-independent:
    τ_r = (1 - 1/(r+1)) / (r/(r+1)) ≡ 1. This is the canonical result —
    resetting does not help exponentials. We check the constancy
    numerically across r in [0.1, 5]."""
    torch.manual_seed(0)
    samples = -torch.log(torch.rand(4096))
    taus = torch.stack(
        [mfpt_with_resetting(samples, r) for r in [0.1, 0.5, 1.0, 2.0, 5.0]]
    )
    spread = (taus.max() - taus.min()).item()
    assert spread < 0.15


def test_coefficient_of_variation_for_exponential_is_one() -> None:
    """CoV of Exponential is exactly 1; sample-CoV should be near 1."""
    torch.manual_seed(0)
    samples = -torch.log(torch.rand(4096))
    cov = coefficient_of_variation(samples).item()
    assert 0.85 < cov < 1.15


def test_coefficient_of_variation_for_heavy_tailed_exceeds_one() -> None:
    """Heavy-tailed first-passage gives CoV > 1 → r* > 0 expected."""
    torch.manual_seed(0)
    # Lognormal has CoV = sqrt(exp(σ²) - 1).
    samples = torch.exp(torch.randn(4096) * 1.0)  # σ = 1, CoV = sqrt(e-1) ≈ 1.31
    cov = coefficient_of_variation(samples).item()
    assert cov > 1.05


def test_optimal_resetting_rate_finds_minimum() -> None:
    """For a heavy-tailed sample, the optimal rate r* is strictly positive
    and τ at r* is at most τ at r → 0."""
    torch.manual_seed(0)
    samples = torch.exp(torch.randn(4096) * 1.0)  # heavy-tailed lognormal
    r_star, tau_star = optimal_resetting_rate(samples, r_min=1e-3, r_max=20.0)
    tau_zero = mfpt_with_resetting(samples, 1e-6).item()
    assert r_star > 0
    assert tau_star <= tau_zero + 1e-2


def test_apply_resetting_fires_with_high_probability() -> None:
    """At very large rate*dt → 1, every sample is reset."""
    torch.manual_seed(0)
    x = torch.randn(64, 3, 8, 8)
    reset_state = torch.full((1, 3, 8, 8), 99.0)
    gen = torch.Generator().manual_seed(0)
    out = apply_resetting(x, reset_state, rate=1e6, dt=1.0, generator=gen)
    # Every batch element should equal 99.
    assert torch.all(out == 99.0)


def test_apply_resetting_at_zero_rate_is_identity() -> None:
    """rate=0 → no resets, output equals input."""
    torch.manual_seed(0)
    x = torch.randn(16, 1, 4, 4)
    reset_state = torch.zeros_like(x)
    out = apply_resetting(x, reset_state, rate=0.0, dt=1.0)
    assert torch.allclose(out, x)


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        mfpt_with_resetting(torch.zeros(2, 2), 1.0)  # not 1-D
    with pytest.raises(ValueError):
        mfpt_with_resetting(torch.zeros(5), -1.0)  # negative rate
    with pytest.raises(ValueError):
        apply_resetting(torch.zeros(2), torch.zeros(2), rate=-1.0, dt=1.0)
    with pytest.raises(ValueError):
        apply_resetting(torch.zeros(2), torch.zeros(2), rate=1.0, dt=0.0)
    with pytest.raises(ValueError):
        coefficient_of_variation(torch.zeros(0))
    with pytest.raises(ValueError):
        optimal_resetting_rate(torch.ones(10), r_min=0.0, r_max=1.0)
