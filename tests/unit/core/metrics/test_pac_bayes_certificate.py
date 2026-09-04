"""Tests for PAC-Bayes generalisation certificates."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.quantitative.pac_bayes_certificate import (
    PACBayesCertificate,
    kl_isotropic_gaussian,
    mcallester_bound,
    pac_bayes_bernstein_bound,
)


def test_kl_zero_when_posterior_equals_prior() -> None:
    """KL(N(0, σ_0² I) || N(0, σ_0² I)) = 0."""
    mu = torch.zeros(10)
    log_var = torch.zeros(10)  # variance = 1 = prior_var
    kl = kl_isotropic_gaussian(mu, log_var, prior_var=1.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_when_posterior_concentrated() -> None:
    """A concentrated posterior (low variance) has KL > 0 relative to a unit prior."""
    mu = torch.zeros(5)
    log_var = torch.full((5,), -4.0)  # variance ≈ 0.018
    kl = kl_isotropic_gaussian(mu, log_var, prior_var=1.0).item()
    assert kl > 0.0


def test_mcallester_bound_monotone_in_empirical_risk() -> None:
    """Bound is monotonically increasing in the empirical risk."""
    a = mcallester_bound(empirical_risk=0.1, kl_divergence=10.0, n_samples=1000)
    b = mcallester_bound(empirical_risk=0.2, kl_divergence=10.0, n_samples=1000)
    assert b > a


def test_mcallester_bound_decreases_with_more_samples() -> None:
    """More training samples → tighter bound."""
    a = mcallester_bound(empirical_risk=0.1, kl_divergence=10.0, n_samples=100)
    b = mcallester_bound(empirical_risk=0.1, kl_divergence=10.0, n_samples=10000)
    assert b < a


def test_bernstein_tighter_than_mcallester_at_low_variance() -> None:
    """When empirical variance is small, Bernstein refinement beats McAllester."""
    risk = 0.1
    kl = 5.0
    n = 500
    mca = mcallester_bound(risk, kl, n, delta=0.05)
    ber = pac_bayes_bernstein_bound(risk, empirical_variance=0.001, kl_divergence=kl,
                                     n_calibration=n, delta=0.05)
    assert ber < mca


def test_pac_bayes_certificate_metric_returns_nonneg_upper_bound() -> None:
    metric = PACBayesCertificate(kl_divergence=5.0, delta=0.05)
    pred = torch.randn(8, 1, 16, 16) * 0.05  # small residual
    target = torch.zeros_like(pred)
    bound = metric(pred, target)
    risk = ((pred - target).pow(2).flatten(start_dim=1).mean(dim=-1)).mean().item()
    assert bound >= risk
    assert bound >= 0.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        kl_isotropic_gaussian(torch.zeros(3), torch.zeros(4))
    with pytest.raises(ValueError):
        kl_isotropic_gaussian(torch.zeros(3), torch.zeros(3), prior_var=-1.0)
    with pytest.raises(ValueError):
        mcallester_bound(0.1, kl_divergence=-1.0, n_samples=10)
    with pytest.raises(ValueError):
        mcallester_bound(0.1, kl_divergence=1.0, n_samples=0)
    with pytest.raises(ValueError):
        mcallester_bound(0.1, kl_divergence=1.0, n_samples=10, delta=0.0)
    with pytest.raises(ValueError):
        pac_bayes_bernstein_bound(0.1, -0.1, 1.0, 10)
    with pytest.raises(ValueError):
        PACBayesCertificate(kl_divergence=-1.0)
    with pytest.raises(ValueError):
        PACBayesCertificate(delta=1.5)
