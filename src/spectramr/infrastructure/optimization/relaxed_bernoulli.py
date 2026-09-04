"""Relaxed-Bernoulli primitive — Gumbel-Sigmoid reparameterisation.

Implements ``CC-2`` from
``TODO/backlog_paradigm_expansion_roadmap.md``. Reused by:

- M3 (LOUPE / PILOT learnable acquisition) — relaxed sampling masks
- M4 (low-rank + sparse for dynamic MRI) — soft component-selection
- Part-I G (BALD active acquisition)

Mathematics
-----------
Given a logit :math:`\\theta`, the *relaxed Bernoulli* with
temperature :math:`\\tau > 0` is

.. math::

    z = \\sigma\\bigl((\\theta + g_1 - g_0) / \\tau\\bigr),
    \\quad g_i \\sim -\\log(-\\log(\\mathrm{Uniform}(0,1))).

In the limit :math:`\\tau \\to 0`, :math:`z \\to \\mathbb{1}[\\sigma(\\theta) > U]`
recovers a hard Bernoulli sample. The reparameterisation makes the
sampling differentiable wrt :math:`\\theta`, which is the property the
learnable-acquisition pipelines rely on.

This implementation prefers ``torch.distributions.RelaxedBernoulli``
when probabilities are needed in :math:`(0, 1)` and falls back to a
hand-rolled Gumbel-Sigmoid for the logit form (numerically more
stable for small / large logits).
"""

from __future__ import annotations

import torch


def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 0.5,
    hard: bool = False,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Sample from a relaxed Bernoulli via the Gumbel-Sigmoid trick.

    Args:
        logits: Real-valued tensor; ``sigmoid(logits)`` is the soft
            probability mass for ``z = 1``.
        tau: Temperature. Smaller → harder; common choices ``0.1`` … ``1.0``.
        hard: If ``True``, returns a binarised sample using the
            straight-through estimator (zero-bias, identity-gradient).
        eps: Floor for log-arguments to keep the Gumbel sampling finite.

    Returns:
        Tensor of the same shape as ``logits``, values in ``[0, 1]``
        (or ``{0, 1}`` if ``hard=True``).

    Raises:
        ValueError: when ``tau <= 0``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    # Relaxed-Bernoulli (BinConcrete) noise is Logistic(0, 1) -- equivalently the
    # difference of two Gumbels, g_1 - g_0. Logistic(0,1) = logit(U) = log U - log(1-U)
    # (variance pi^2/3). NOTE: log(u1) - log(u2) is the difference of two -Exp(1)
    # variates (Laplace, variance 2.0), the WRONG distribution -- it biased the
    # tau->0 marginal away from sigmoid(logits). Use a single logistic draw instead.
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    g = torch.log(u) - torch.log1p(-u)  # Logistic(0, 1) noise
    soft = torch.sigmoid((logits + g) / tau)
    if not hard:
        return soft
    # Straight-through: forward sample is the binarised value, gradient
    # flows through the soft sample.
    binarised = (soft > 0.5).to(soft.dtype)
    return binarised + (soft - soft.detach())


def relaxed_bernoulli(
    probs: torch.Tensor,
    tau: float = 0.5,
    hard: bool = False,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Relaxed Bernoulli sample given *probabilities* (not logits).

    Convenience wrapper that converts ``probs ∈ (0, 1)`` to logits and
    forwards to :func:`gumbel_sigmoid`. Useful when a calling pipeline
    already maintains a probability tensor (e.g. learned mask).

    Raises:
        ValueError: when ``probs`` falls outside ``[0, 1]`` or ``tau <= 0``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must lie in [0, 1]")
    safe = probs.clamp(eps, 1.0 - eps)
    logits = torch.log(safe) - torch.log1p(-safe)
    return gumbel_sigmoid(logits, tau=tau, hard=hard, eps=eps)


def expected_density(logits: torch.Tensor) -> torch.Tensor:
    """Expected sampling density: ``mean(sigmoid(logits))``.

    Used by LOUPE / PILOT-style penalties that constrain the relaxed-
    Bernoulli mask to a target sparsity. Exposed here so all consumers
    use the same formula.
    """
    return torch.sigmoid(logits).mean()


def density_penalty(
    logits: torch.Tensor,
    target: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Squared deviation from a target sampling density.

    .. math::

        \\mathcal{L}_{density} =
            \\lambda \\bigl(\\mathbb{E}[\\sigma(\\theta)] - \\rho\\bigr)^2

    Args:
        logits: Per-element logit tensor.
        target: Target density ``ρ ∈ (0, 1)``.
        weight: Loss multiplier ``λ ≥ 0``.

    Raises:
        ValueError: when ``target ∉ (0, 1)`` or ``weight < 0``.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must lie in (0, 1), got {target}")
    if weight < 0:
        raise ValueError(f"weight must be ≥ 0, got {weight}")
    return weight * (expected_density(logits) - target).pow(2)


__all__ = [
    "density_penalty",
    "expected_density",
    "gumbel_sigmoid",
    "relaxed_bernoulli",
]
