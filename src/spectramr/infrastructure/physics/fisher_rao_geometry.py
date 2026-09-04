r"""Fisher-Rao information geometry for Bloch-likelihood flow matching.

For a parametric family :math:`\{p(\mathbf{y}\mid\theta;B_0)\}` of MRI
acquisition likelihoods, the *Fisher information matrix* is

.. math::

    g_{ij}(\theta;B_0) = \mathbb{E}_{\mathbf{y}\mid\theta}
        \!\big[\partial_{\theta^i}\log p\;\partial_{\theta^j}\log p\big].

Under additive Gaussian noise :math:`\mathbf{y}=\mathbf{s}(\theta;B_0)+\mathbf{n}`,
:math:`\mathbf{n}\sim\mathcal{N}(0,\sigma^2 I)`, this collapses to the
Bloch-Jacobian Gram matrix

.. math::

    g_{ij} = \frac{1}{\sigma^2}\,\bigl(\partial_{\theta^i}\mathbf{s}\bigr)^{\!\top}\!
              \bigl(\partial_{\theta^j}\mathbf{s}\bigr),

which is the unique :math:`\mathrm{GL}`-invariant metric on the statistical
manifold :math:`\mathcal{P}(\Theta)` [Amari-Nagaoka 2000].

The Fisher-Rao geodesic between :math:`\pi_0` and :math:`\pi_1` minimises

.. math::

    \mathrm{FR}^2(\pi_0,\pi_1) = \inf_{(\rho_t,v_t)}\!\int_0^1\!\!\int_\Theta
        v_t(\theta)^{\!\top}\,g(\theta)\,v_t(\theta)\,\rho_t(\theta)\,d\theta\,dt,

subject to the continuity equation :math:`\partial_t\rho_t+\nabla\!\cdot(\rho_t v_t)=0`.
Flow-matching with the Fisher-Rao metric yields ULF→HF translation
maps that are *intrinsic to the Bloch likelihood* rather than to an ad-hoc
image-space loss.

The module exports:

* :func:`fisher_information_from_jacobian` — closed-form Gaussian-noise
  Fisher matrix :math:`g_{ij}` from the Bloch Jacobian.
* :func:`fisher_norm` — :math:`\|v\|_g=\sqrt{v^{\top}g\,v}`.
* :func:`monte_carlo_fisher_estimator` — :math:`O(1/\sqrt M)` MC estimator
  with variance bound (Mingo-Speicher rate).
* :func:`fisher_rao_velocity_loss` — flow-matching training loss
  :math:`\|u_\phi(\theta,t)-v_t^\star(\theta)\|_g^2`.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "fisher_information_from_jacobian",
    "fisher_norm",
    "fisher_rao_velocity_loss",
    "monte_carlo_fisher_estimator",
]


def fisher_information_from_jacobian(jacobian: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r"""Closed-form Gaussian-likelihood Fisher matrix
    :math:`g=J^{\top}J/\sigma^2`.

    Args:
        jacobian: ``(..., N, d)`` Bloch-signal Jacobian :math:`\partial\mathbf{s}/\partial\theta`,
            where :math:`N` is the number of readouts and :math:`d` the
            parameter dimension.
        sigma: Gaussian noise standard deviation. Must be positive.

    Returns:
        ``(..., d, d)`` Fisher information matrix, positive-semi-definite.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}.")
    if jacobian.ndim < 2:
        raise ValueError(
            f"jacobian must have rank >= 2 with last two axes (N, d); got {tuple(jacobian.shape)}."
        )
    return torch.einsum("...nd,...ne->...de", jacobian, jacobian) / (sigma**2)


def fisher_norm(vectors: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    r"""Fisher-Rao norm :math:`\|v\|_g=\sqrt{v^{\top}g\,v}`.

    Args:
        vectors: ``(..., d)`` tangent vectors.
        metric: ``(..., d, d)`` Fisher matrix; must be broadcastable with
            ``vectors`` along leading axes.

    Returns:
        ``(...,)`` norm magnitudes (non-negative).
    """
    if vectors.shape[-1] != metric.shape[-1] or metric.shape[-1] != metric.shape[-2]:
        raise ValueError(
            f"shape mismatch: vectors {tuple(vectors.shape)}, metric {tuple(metric.shape)}."
        )
    quad = torch.einsum("...d,...de,...e->...", vectors, metric, vectors)
    return torch.sqrt(quad.clamp_min(0.0))


def monte_carlo_fisher_estimator(
    jacobian_samples: torch.Tensor, sigma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Empirical Fisher matrix from :math:`M` Bloch-Jacobian samples.

    .. math::

        \hat g = \frac{1}{M\sigma^2}\sum_{m=1}^{M} J_m^{\top}J_m,

    with the per-sample-Frobenius-variance bound
    :math:`\mathbb{E}\|\hat g-g\|_F^2\le\mathrm{tr}(g)^2\,\kappa(g)/M`
    (Mingo-Speicher 2017, Cor. 5.10) where :math:`\kappa` is the condition number.

    Args:
        jacobian_samples: ``(M, N, d)`` stack of Bloch Jacobians at the same
            tissue parameters across :math:`M` noise realisations.
        sigma: Gaussian noise standard deviation.

    Returns:
        ``(g_hat, std_err)`` where ``g_hat`` is the empirical mean Fisher
        matrix and ``std_err`` is the per-entry standard error
        :math:`O(1/\sqrt M)`.
    """
    if jacobian_samples.ndim != 3:
        raise ValueError(
            f"jacobian_samples must be (M, N, d); got {tuple(jacobian_samples.shape)}."
        )
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    per_sample = torch.einsum("mnd,mne->mde", jacobian_samples, jacobian_samples) / (sigma**2)
    g_hat = per_sample.mean(dim=0)
    std_err = per_sample.std(dim=0, unbiased=False) / math.sqrt(jacobian_samples.shape[0])
    return g_hat, std_err


def fisher_rao_velocity_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    metric: torch.Tensor,
) -> torch.Tensor:
    r"""Fisher-Rao flow-matching loss
    :math:`\|u_\phi(\theta,t)-v_t^\star(\theta)\|_g^2`.

    Args:
        predicted_velocity: ``(..., d)`` learned velocity field.
        target_velocity: ``(..., d)`` ground-truth Fisher-Rao geodesic
            velocity.
        metric: ``(..., d, d)`` Fisher matrix at the sampling points.

    Returns:
        Scalar loss (mean over leading batch axes).
    """
    diff = predicted_velocity - target_velocity
    quad = torch.einsum("...d,...de,...e->...", diff, metric, diff)
    return quad.clamp_min(0.0).mean()
