"""Tests for the relaxed-Bernoulli (Gumbel-Sigmoid) primitive.

Targets ``mriforge.infrastructure.optimization.relaxed_bernoulli`` —
``CC-2`` from ``TODO/backlog_paradigm_expansion_roadmap.md``. Used by
M3 (LOUPE / PILOT), M4 (low-rank + sparse), and Part-I G (BALD).

Categories:

- ``gumbel_sigmoid`` returns a ``[0, 1]`` tensor of the same shape
- ``hard=True`` produces a binary tensor with straight-through gradient
- ``relaxed_bernoulli`` rejects out-of-range probabilities
- ``tau ≤ 0`` raises
- Mean of many soft samples ≈ ``sigmoid(logits)``
- ``expected_density`` matches the Bernoulli mean
- ``density_penalty`` is zero when expected density equals the target
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.optimization.relaxed_bernoulli import (
    density_penalty,
    expected_density,
    gumbel_sigmoid,
    relaxed_bernoulli,
)


# ---------------------------------------------------------------------------
# gumbel_sigmoid
# ---------------------------------------------------------------------------


def test_gumbel_sigmoid_shape_preserved() -> None:
    """Output shape matches input."""
    logits = torch.zeros(4, 8)
    out = gumbel_sigmoid(logits, tau=0.5)
    assert out.shape == logits.shape


def test_gumbel_sigmoid_in_unit_interval() -> None:
    """Soft samples are bounded in ``[0, 1]``."""
    logits = torch.randn(64)
    out = gumbel_sigmoid(logits, tau=0.5)
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_gumbel_sigmoid_hard_is_binary() -> None:
    """``hard=True`` returns exactly 0 or 1 in the forward pass."""
    logits = torch.randn(128)
    out = gumbel_sigmoid(logits, tau=0.5, hard=True)
    unique = torch.unique(out)
    assert torch.all((unique == 0) | (unique == 1))


def test_gumbel_sigmoid_invalid_tau_raises() -> None:
    """``tau ≤ 0`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="tau"):
        gumbel_sigmoid(torch.zeros(4), tau=0.0)


def test_gumbel_sigmoid_gradient_flows() -> None:
    """The relaxed sample is differentiable wrt the logits."""
    logits = torch.randn(8, requires_grad=True)
    out = gumbel_sigmoid(logits, tau=0.5)
    out.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gumbel_sigmoid_hard_straight_through_gradient() -> None:
    """Even with ``hard=True`` the straight-through trick passes gradient."""
    logits = torch.randn(8, requires_grad=True)
    out = gumbel_sigmoid(logits, tau=0.5, hard=True)
    out.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gumbel_sigmoid_mean_matches_sigmoid() -> None:
    """Averaged over many samples, the soft output approaches ``sigmoid(logits)``.

    A loose tolerance is fine; the test is about *unbiasedness*, not speed.
    The SOFT sample is only exactly unbiased in the tau->0 limit (the
    hard-sample marginal equals ``sigmoid`` exactly -- covered by
    ``test_gumbel_sigmoid_hard_marginal_matches_sigmoid``); at finite tau there
    is a small bias toward 0.5, so use a low tau where the approximation is tight.
    """
    torch.manual_seed(0)
    logits = torch.tensor([-1.0, 0.0, 1.0])
    n_samples = 4096
    samples = torch.stack(
        [gumbel_sigmoid(logits, tau=0.3) for _ in range(n_samples)]
    )
    mean = samples.mean(dim=0)
    assert torch.allclose(mean, torch.sigmoid(logits), atol=0.05)


# ---------------------------------------------------------------------------
# relaxed_bernoulli (probability variant)
# ---------------------------------------------------------------------------


def test_relaxed_bernoulli_basic() -> None:
    """Probabilities pathway returns a same-shape tensor in ``[0, 1]``."""
    probs = torch.tensor([0.1, 0.5, 0.9])
    out = relaxed_bernoulli(probs, tau=0.5)
    assert out.shape == probs.shape
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_relaxed_bernoulli_negative_probs_rejected() -> None:
    """Out-of-range probabilities → ``ValueError``."""
    with pytest.raises(ValueError, match=r"probs"):
        relaxed_bernoulli(torch.tensor([-0.1, 0.5]), tau=0.5)


def test_relaxed_bernoulli_above_one_rejected() -> None:
    """``probs > 1`` rejected."""
    with pytest.raises(ValueError, match=r"probs"):
        relaxed_bernoulli(torch.tensor([0.5, 1.5]), tau=0.5)


# ---------------------------------------------------------------------------
# expected_density + density_penalty
# ---------------------------------------------------------------------------


def test_expected_density_matches_sigmoid_mean() -> None:
    """``expected_density`` is just ``sigmoid(logits).mean()``."""
    logits = torch.tensor([-2.0, 0.0, 2.0])
    expected = torch.sigmoid(logits).mean()
    assert pytest.approx(expected_density(logits).item()) == expected.item()


def test_density_penalty_zero_at_target() -> None:
    """Penalty is zero when the expected density equals the target."""
    # sigmoid(0) = 0.5
    logits = torch.zeros(8)
    pen = density_penalty(logits, target=0.5, weight=1.0)
    assert pytest.approx(pen.item(), abs=1e-7) == 0.0


def test_density_penalty_grows_with_deviation() -> None:
    """Larger deviation from the target → larger penalty."""
    logits = torch.zeros(8)  # density = 0.5
    near = density_penalty(logits, target=0.4, weight=1.0).item()
    far = density_penalty(logits, target=0.1, weight=1.0).item()
    assert far > near


def test_density_penalty_invalid_target_raises() -> None:
    """``target ∉ (0, 1)`` rejected."""
    with pytest.raises(ValueError, match="target"):
        density_penalty(torch.zeros(4), target=1.5)


def test_density_penalty_negative_weight_raises() -> None:
    """``weight < 0`` rejected."""
    with pytest.raises(ValueError, match="weight"):
        density_penalty(torch.zeros(4), target=0.5, weight=-1.0)


# ---------------------------------------------------------------------------
# Noise distribution (regression — Laplace-vs-Logistic noise, 2026-06-12 audit)
# ---------------------------------------------------------------------------


def test_gumbel_sigmoid_hard_marginal_matches_sigmoid() -> None:
    """``P(z = 1)`` of a hard relaxed-Bernoulli sample must equal
    ``sigmoid(logits)``.

    Regression for the wrong reparameterisation noise: ``log(u1) - log(u2)``
    is a difference of two ``-Exp(1)`` variates (Laplace, variance 2.0), not
    the Logistic(0,1) (variance pi^2/3) required by the relaxed Bernoulli, so
    the tau->0 marginal was biased (~0.752 instead of sigmoid(0.7)=0.668).
    """
    torch.manual_seed(0)
    logit_val = 0.7
    logits = torch.full((200_000,), logit_val)
    samples = gumbel_sigmoid(logits, tau=0.5, hard=True)
    target = torch.sigmoid(torch.tensor(logit_val)).item()
    assert abs(samples.mean().item() - target) < 0.01


def test_gumbel_sigmoid_noise_is_logistic_variance() -> None:
    """The injected reparameterisation noise has Logistic variance pi^2/3."""
    torch.manual_seed(1)
    # tau=1, logits=0 -> soft = sigmoid(noise); recover noise via logit(soft).
    logits = torch.zeros(500_000)
    soft = gumbel_sigmoid(logits, tau=1.0, hard=False).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(soft) - torch.log1p(-soft)
    assert abs(noise.var().item() - math.pi**2 / 3.0) < 0.1
