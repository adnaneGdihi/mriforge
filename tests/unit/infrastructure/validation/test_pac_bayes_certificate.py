"""Tests for the PAC-Bayes McAllester certifier.

Focus: the bounded-loss precondition fix (2026-06-12 validation audit). The
McAllester bound is only valid for a loss in ``[0, M]``; the certifier must
RAISE on an out-of-range empirical risk rather than emit a meaningless number,
and must rescale correctly when ``M != 1``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.validation.pac_bayes_certificate import (
    PACBayesCertifier,
    gaussian_kl_diagonal,
    mcallester_bound,
)


def _tiny_model() -> nn.Module:
    m = nn.Linear(4, 2)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()  # KL is finite and small for zero weights
    return m


def test_mcallester_bound_infinite_for_single_sample():
    assert mcallester_bound(0.1, kl=1.0, n=1) == float("inf")


def test_mcallester_bound_exceeds_empirical_risk():
    # The slack term is non-negative, so the bound is always >= empirical risk.
    b = mcallester_bound(0.2, kl=2.0, n=100, delta=0.05)
    assert b >= 0.2
    assert math.isfinite(b)


def test_gaussian_kl_zero_mean_unit_consistency():
    # KL(N(0, σ_p²) ‖ N(0, σ_p²)) over k params == 0 when posterior == prior.
    prior_sigma = 0.1
    mean = torch.zeros(5)
    logvar = torch.log(torch.tensor(prior_sigma**2)).expand_as(mean)
    kl = float(gaussian_kl_diagonal(mean, logvar, prior_sigma))
    assert abs(kl) < 1e-5


def test_certify_raises_on_unbounded_loss():
    cert = PACBayesCertifier(prior_sigma=0.1, delta=0.05)
    # A raw-MSE-style loss of 5.0 violates the [0, 1] precondition.
    with pytest.raises(ValueError, match=r"requires a loss in \[0, 1"):
        cert.certify(_tiny_model(), loss_fn=lambda _m: 5.0, n_samples=64)


def test_certify_raises_on_negative_loss():
    cert = PACBayesCertifier()
    with pytest.raises(ValueError):
        cert.certify(_tiny_model(), loss_fn=lambda _m: -0.1, n_samples=64)


def test_certify_accepts_bounded_loss():
    cert = PACBayesCertifier(prior_sigma=0.1, delta=0.05)
    out = cert.certify(_tiny_model(), loss_fn=lambda _m: 0.3, n_samples=128)
    assert out.empirical_risk == 0.3
    assert math.isfinite(out.bound)
    assert out.bound >= out.empirical_risk


def test_certify_rescales_for_nonunit_loss_bound():
    cert = PACBayesCertifier(prior_sigma=0.1, delta=0.05)
    model = _tiny_model()
    # With loss_bound=2 and risk=1.5, the bound is 2 * mcallester(0.75, ...).
    out = cert.certify(
        model, loss_fn=lambda _m: 1.5, n_samples=128, loss_bound=2.0
    )
    kl = cert.kl_for_model(model)
    expected = 2.0 * mcallester_bound(0.75, kl, 128, 0.05)
    assert abs(out.bound - expected) < 1e-9


def test_certify_rejects_nonpositive_loss_bound():
    cert = PACBayesCertifier()
    with pytest.raises(ValueError, match="loss_bound must be"):
        cert.certify(_tiny_model(), loss_fn=lambda _m: 0.5, n_samples=8, loss_bound=0.0)
