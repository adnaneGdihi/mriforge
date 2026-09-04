"""Tests for Pearl front-door causal identification."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.causal_frontdoor import (
    approximate_frontdoor_bound,
    donsker_varadhan_mi,
    frontdoor_adjusted_prediction,
    frontdoor_loss,
)


def test_donsker_varadhan_zero_for_uninformative_critic() -> None:
    """T_η ≡ const → MI bound is 0."""
    n = 64
    joint = torch.zeros(n)
    marginal = torch.zeros(n)
    mi = donsker_varadhan_mi(joint, marginal).item()
    assert abs(mi) < 1e-5


def test_donsker_varadhan_positive_for_informative_critic() -> None:
    """Higher score on joint than marginal → positive MI lower bound."""
    torch.manual_seed(0)
    joint = torch.full((64,), 0.5)
    marginal = torch.full((64,), -0.5)
    mi = donsker_varadhan_mi(joint, marginal).item()
    assert mi > 0.0


def test_frontdoor_adjusted_distribution_is_probability_per_a() -> None:
    """Each column of P(I|do(A)) is a probability distribution."""
    # |M|=3, |A|=4, |I|=5.
    p_m_given_a = torch.softmax(torch.randn(3, 4), dim=0)
    p_i_given_m_a = torch.softmax(torch.randn(5, 3, 4), dim=0)
    p_a = torch.softmax(torch.randn(4), dim=0)
    out = frontdoor_adjusted_prediction(p_m_given_a, p_i_given_m_a, p_a)
    assert out.shape == (5, 4)
    # Each column sums to 1.
    sums = out.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_frontdoor_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        frontdoor_adjusted_prediction(torch.zeros(3), torch.zeros(5, 3, 4), torch.zeros(4))
    with pytest.raises(ValueError):
        frontdoor_adjusted_prediction(
            torch.zeros(3, 4), torch.zeros(5, 2, 4), torch.zeros(4)  # |M| mismatch
        )


def test_frontdoor_loss_zero_when_predictions_match_and_no_mi() -> None:
    x = torch.randn(8, 3, 8, 8)
    mi = torch.tensor(0.0)
    loss = frontdoor_loss(x, x, mi, mi_weight=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_frontdoor_loss_positive_with_mi() -> None:
    x = torch.zeros(8, 1)
    mi = torch.tensor(0.5)
    loss = frontdoor_loss(x, x, mi, mi_weight=2.0)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_approximate_frontdoor_bound_formula() -> None:
    """TV bound = √(ε/2) + 1/√N."""
    bound = approximate_frontdoor_bound(mi_estimate=0.08, n_samples=400)
    expected = math.sqrt(0.04) + 0.05
    assert bound == pytest.approx(expected, abs=1e-6)


def test_approximate_frontdoor_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        approximate_frontdoor_bound(mi_estimate=-0.1, n_samples=100)
    with pytest.raises(ValueError):
        approximate_frontdoor_bound(mi_estimate=0.1, n_samples=0)
    with pytest.raises(ValueError):
        frontdoor_loss(torch.zeros(3), torch.zeros(3), torch.tensor(0.0), mi_weight=-1.0)
