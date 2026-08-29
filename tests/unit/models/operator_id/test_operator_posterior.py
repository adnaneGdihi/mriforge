r"""Tests for the Laplace operator posterior (SOTA plan T3).

T3 replaces blind JSMoCo with a *calibrated* uncertainty posterior over the BCH
operator parameters (severities): a Laplace approximation
``Σ = (H + λ₀I)⁻¹`` with ``H`` the Hessian of the negative log-posterior at the
MAP. Corner case: ``λ₀ → ∞`` (or infinite curvature) ⇒ ``Σ → 0`` ⇒ the BCH point
estimate (the existing incumbent), so the posterior provably generalises it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.models.operator_id.operator_posterior import (  # noqa: E402
    laplace_posterior,
    operator_posterior_std,
    sample_operators,
)


def _quadratic_nll(curv):
    """neg-log-post = 0.5 · θ·diag(curv)·θ → Hessian = diag(curv)."""

    def fn(theta):
        return 0.5 * (theta * theta * curv).sum()

    return fn


def test_laplace_covariance_is_inverse_hessian():
    curv = torch.tensor([1.0, 4.0, 9.0])
    theta = torch.zeros(3)
    mean, cov = laplace_posterior(theta, _quadratic_nll(curv))
    assert torch.allclose(mean, theta)
    # cov = H⁻¹ = diag(1/curv)
    assert torch.allclose(torch.diagonal(cov), 1.0 / curv, atol=1e-5)


def test_posterior_std_matches_inverse_curvature():
    curv = torch.tensor([1.0, 4.0, 16.0])
    _, cov = laplace_posterior(torch.zeros(3), _quadratic_nll(curv))
    std = operator_posterior_std(cov)
    assert torch.allclose(std, 1.0 / curv.sqrt(), atol=1e-5)  # [1, 0.5, 0.25]


def test_high_prior_precision_collapses_to_point_estimate():
    """Corner case: λ₀ → ∞ ⇒ Σ → 0 ⇒ BCH point estimate."""
    curv = torch.tensor([1.0, 4.0, 9.0])
    _, cov = laplace_posterior(torch.zeros(3), _quadratic_nll(curv), prior_precision=1e8)
    std = operator_posterior_std(cov)
    assert torch.all(std < 1e-3)
    # samples collapse onto the mean
    mean = torch.tensor([0.5, 1.0, 2.0])
    s = sample_operators(mean, cov, n_samples=16, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s, mean.expand_as(s), atol=1e-2)


def test_sample_shape_and_distribution():
    curv = torch.tensor([1.0, 1.0])
    mean = torch.tensor([2.0, -1.0])
    _, cov = laplace_posterior(torch.zeros(2), _quadratic_nll(curv))
    s = sample_operators(mean, cov, n_samples=4000, generator=torch.Generator().manual_seed(0))
    assert s.shape == (4000, 2)
    assert torch.allclose(s.mean(0), mean, atol=0.1)
    assert torch.allclose(s.std(0), torch.ones(2), atol=0.15)  # unit curvature → unit std


def test_rejects_negative_prior_precision():
    with pytest.raises(ValueError):
        laplace_posterior(torch.zeros(2), _quadratic_nll(torch.ones(2)), prior_precision=-1.0)
