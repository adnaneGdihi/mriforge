r"""Pearl front-door causal identification for scanner-confounded MRI.

When scanner identity :math:`S` confounds anatomy :math:`A` and image
:math:`I` but is *unmeasured at inference* (a new site, an out-of-domain
vendor), back-door adjustment fails. The **front-door criterion**
[Pearl 2009, Th. 3.3.4] identifies :math:`P(I\mid\mathrm{do}(A))` through a
mediator :math:`M` (or :math:`K`, the k-space spectral profile in
Batch-2 #10) that intercepts every directed path :math:`A\to I`:

.. math::

    P(I=i\mid\mathrm{do}(A=a))
        = \sum_m P(M=m\mid A=a)
            \sum_{a'} P(I=i\mid M=m,A=a')\,P(A=a').

Approximate front-door identification holds whenever the mediator is
*site-invariant* — i.e. the mutual information :math:`\mathrm{MI}(M;S)`
is small. Donsker-Varadhan + Pinsker give the total-variation bound

.. math::

    \mathrm{TV}(\hat P, P(\cdot\mid\mathrm{do}(\cdot))) \le \sqrt{\varepsilon/2} + O(N^{-1/2}),

with :math:`\varepsilon = \widehat{\mathrm{MI}}(M;S)`.

Exports:

* :func:`donsker_varadhan_mi` — neural-MI estimator surrogate
  :math:`\sup_T [\mathbb{E}_{P_{XY}}T-\log\mathbb{E}_{P_X\otimes P_Y}e^T]`.
* :func:`frontdoor_adjusted_prediction` — evaluate (16) given empirical
  :math:`P(M\mid A)` and :math:`P(I\mid M,S)`.
* :func:`frontdoor_loss` — reconstruction MSE + :math:`\lambda\,\widehat{\mathrm{MI}}(M;S)`.
* :func:`approximate_frontdoor_bound` — closed-form
  :math:`\sqrt{\varepsilon/2}+O(N^{-1/2})` TV upper bound.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "approximate_frontdoor_bound",
    "donsker_varadhan_mi",
    "frontdoor_adjusted_prediction",
    "frontdoor_loss",
]


def donsker_varadhan_mi(
    joint_scores: torch.Tensor,
    marginal_scores: torch.Tensor,
) -> torch.Tensor:
    r"""Donsker-Varadhan lower bound on the mutual information.

    .. math::

        \mathrm{MI}(M;S) \ge \mathbb{E}_{P_{MS}}\!\big[T_\eta(M,S)\big]
            - \log\mathbb{E}_{P_M\otimes P_S}\!\big[e^{T_\eta(M,S)}\big].

    Args:
        joint_scores: ``(N,)`` :math:`T_\eta(M_i,S_i)` evaluated on the
            paired joint samples.
        marginal_scores: ``(N,)`` :math:`T_\eta(M_i,S_j)` evaluated on
            independent product-marginal samples.

    Returns:
        Scalar :math:`\hat I_\eta` lower bound.
    """
    if joint_scores.ndim != 1 or marginal_scores.ndim != 1:
        raise ValueError("joint_scores and marginal_scores must be 1-D.")
    return joint_scores.mean() - torch.logsumexp(
        marginal_scores - math.log(marginal_scores.shape[0]), dim=0
    )


def frontdoor_adjusted_prediction(
    p_m_given_a: torch.Tensor,
    p_i_given_m_a: torch.Tensor,
    p_a: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate the front-door formula
    :math:`P(I\mid\mathrm{do}(A=a))=\sum_m P(M\mid A)\sum_{a'} P(I\mid M, A')P(A')`.

    Args:
        p_m_given_a: ``(|M|, |A|)`` conditional :math:`P(M\mid A)`.
        p_i_given_m_a: ``(|I|, |M|, |A|)`` conditional :math:`P(I\mid M, A)`.
        p_a: ``(|A|,)`` marginal :math:`P(A)`.

    Returns:
        ``(|I|, |A|)`` matrix :math:`P(I\mid\mathrm{do}(A))`.
    """
    if p_m_given_a.ndim != 2 or p_i_given_m_a.ndim != 3 or p_a.ndim != 1:
        raise ValueError("shape mismatch.")
    if p_m_given_a.shape[0] != p_i_given_m_a.shape[1]:
        raise ValueError("|M| must match across p_m_given_a and p_i_given_m_a.")
    if p_a.shape[0] != p_m_given_a.shape[1]:
        raise ValueError("|A| must match.")
    if p_a.shape[0] != p_i_given_m_a.shape[2]:
        raise ValueError("|A| must match across all inputs.")
    # Inner sum: P(I|M) = Σ_a' P(I|M, a') P(a').
    p_i_given_m = torch.einsum("ima,a->im", p_i_given_m_a, p_a)
    # Outer sum: P(I|do(A)) = Σ_m P(I|M=m) P(M=m|A).
    return p_i_given_m @ p_m_given_a


def frontdoor_loss(
    predicted_image: torch.Tensor,
    target_image: torch.Tensor,
    mi_estimate: torch.Tensor,
    mi_weight: float = 1.0,
) -> torch.Tensor:
    r"""Front-door training loss = reconstruction MSE + :math:`\lambda\,\widehat{\mathrm{MI}}(M;S)`.

    Args:
        predicted_image: ``(B, ...)`` model prediction.
        target_image: ``(B, ...)`` ground truth.
        mi_estimate: Scalar :math:`\widehat{\mathrm{MI}}(M;S)\ge 0`. Should
            already be detached or computed differentiably depending on
            whether the MI is being adversarially optimised.
        mi_weight: :math:`\lambda\ge 0` controlling the trade-off.

    Returns:
        Scalar loss.
    """
    if mi_weight < 0:
        raise ValueError("mi_weight must be non-negative.")
    recon = (predicted_image - target_image).pow(2).mean()
    return recon + mi_weight * mi_estimate.clamp_min(0.0)


def approximate_frontdoor_bound(mi_estimate: float, n_samples: int) -> float:
    r"""TV bound :math:`\sqrt{\varepsilon/2}+O(N^{-1/2})` from Pinsker + DV.

    Quantifies the cost of approximate front-door identification when the
    mediator is *almost* but not exactly site-invariant.

    Args:
        mi_estimate: Empirical mediator-scanner MI :math:`\widehat{\mathrm{MI}}(M;S)\ge 0`.
        n_samples: Number of paired samples used for the DV estimator.

    Returns:
        Total-variation upper bound between the front-door estimator and
        the true causal effect.
    """
    if mi_estimate < 0:
        raise ValueError("mi_estimate must be non-negative.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return math.sqrt(mi_estimate / 2.0) + 1.0 / math.sqrt(n_samples)
