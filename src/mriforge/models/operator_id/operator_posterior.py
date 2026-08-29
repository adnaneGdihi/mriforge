r"""Laplace posterior over BCH operator parameters (SOTA plan T3).

T3 replaces the REJECTed blind operator-image diffusion (JSMoCo) — non-identifiable
on magnitude data, with an image prior that absorbs operator error — by a
*physics-parametric* operator identification (the BCH ``operator_id`` incumbent,
:class:`~mriforge.models.operator_id.bch_conditioner.BCHOperatorConditioner`)
equipped with a **calibrated uncertainty posterior** over the operator severities.

The posterior is a Laplace approximation at the MAP severities :math:`\hat{s}`:

.. math::
    \Sigma = \big(H + \lambda_0 I\big)^{-1}, \qquad
    H = \nabla^2_{s}\, \big[-\log p(s\mid y)\big]\big|_{\hat{s}},

with ``H`` the Hessian of the (structured-Gaussian) negative log-posterior and
:math:`\lambda_0` a Gaussian-prior precision. **Corner case:**
:math:`\lambda_0 \to \infty` (or infinite likelihood curvature) ⇒ :math:`\Sigma \to 0`
⇒ the BCH point estimate (the incumbent), so the posterior provably generalises it.

**Identifiability requirement (honest):** a meaningful operator posterior needs
**complex multi-coil k-space** (Ahmed–Recht–Romberg bilinear identifiability); on
single-contrast magnitude data the operator is non-identifiable and the posterior
is uninformative. Position against C-MSM [arXiv:2509.18402].
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["laplace_posterior", "operator_posterior_std", "sample_operators"]

_JITTER = 1e-8


def laplace_posterior(
    params: torch.Tensor,
    neg_log_post_fn: Callable[[torch.Tensor], torch.Tensor],
    prior_precision: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Laplace posterior ``(mean, cov)`` over a fitted parameter vector.

    Args:
        params: the MAP parameters :math:`\hat{s}` (``[D]``).
        neg_log_post_fn: scalar negative-log-posterior as a function of ``params``.
        prior_precision: isotropic Gaussian-prior precision :math:`\lambda_0 \ge 0`.

    Returns:
        ``(mean, cov)`` with ``mean = params`` and ``cov = (H + λ₀I)⁻¹``
        (symmetrised). As ``λ₀ → ∞`` the covariance collapses to 0 (point estimate).
    """
    if prior_precision < 0:
        raise ValueError(f"prior_precision must be >= 0, got {prior_precision}")
    theta = params.detach()
    d = theta.numel()
    hess = torch.autograd.functional.hessian(neg_log_post_fn, theta)
    hess = hess.reshape(d, d)
    hess = 0.5 * (hess + hess.transpose(-1, -2))  # symmetrise numerical asymmetry
    eye = torch.eye(d, dtype=hess.dtype, device=hess.device)
    precision = hess + prior_precision * eye
    cov = torch.linalg.inv(precision + _JITTER * eye)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return theta, cov


def operator_posterior_std(cov: torch.Tensor) -> torch.Tensor:
    """Per-parameter posterior standard deviation ``√diag(Σ)`` (clamped ≥ 0)."""
    return torch.sqrt(torch.diagonal(cov).clamp(min=0.0))


def sample_operators(
    mean: torch.Tensor,
    cov: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Draw ``n_samples`` operator-parameter vectors ``~ N(mean, cov)`` (``[n, D]``).

    Uses a jittered Cholesky factor; with ``cov → 0`` (high prior precision) all
    samples collapse onto ``mean`` (the point estimate).
    """
    d = mean.numel()
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    chol = torch.linalg.cholesky(cov + _JITTER * eye)
    z = torch.randn(n_samples, d, generator=generator, dtype=mean.dtype, device=mean.device)
    return mean.unsqueeze(0) + z @ chol.transpose(-1, -2)
