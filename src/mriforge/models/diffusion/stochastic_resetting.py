r"""Stochastic-resetting modifier for score-based reverse diffusion.

The standard reverse SDE

.. math::

    d\mathbf{x}_t = \big[-f(\mathbf{x}_t,t)+g(t)^2\,s_\theta(\mathbf{x}_t,t)\big]\,dt
                    + g(t)\,d\bar W_t

can be accelerated by intermittently *resetting* the sample back to a
data-consistent state. Following Evans & Majumdar (2011, 2020), the
resetting-modified SDE

.. math::

    d\mathbf{x}_t = [\dots]\,dt + g\,d\bar W_t
                    - \mathbb{1}[\xi_t\le r\,dt]\,(\mathbf{x}_t-\mathcal{R}(\hat{\mathbf{y}})),

with :math:`\xi_t\sim\mathcal{U}[0,1]` and resetting kernel
:math:`\mathcal{R}(\hat{\mathbf{y}})=S^{\!\dagger}F^{\!\dagger}\hat{\mathbf{y}}`,
satisfies the mean-first-passage-time identity

.. math::

    \tau_r = \frac{1-\tilde P(r)}{r\,\tilde P(r)},

where :math:`\tilde P(r)=\mathbb{E}[e^{-r\tau_0}]` is the Laplace transform
of the unreset first-passage density. Whenever the coefficient of variation
:math:`\mathrm{CoV}(\tau_0)=\sqrt{\mathrm{Var}(\tau_0)}/\mathbb{E}[\tau_0]>1`,
the optimal resetting rate :math:`r^\star=\arg\min_r\tau_r` is strictly
positive and the reset SDE *strictly accelerates* the unreset one.

This module exports:

* :func:`mfpt_with_resetting` — the closed-form
  :math:`\tau_r=(1-\tilde P(r))/(r\tilde P(r))`.
* :func:`optimal_resetting_rate` — numerical :math:`r^\star` via convex
  one-dimensional search.
* :func:`apply_resetting` — Bernoulli-mask one reverse-SDE step under a
  resetting rate.
* :func:`coefficient_of_variation` — :math:`\sqrt{\mathrm{Var}(\tau)}/\mathbb{E}[\tau]`.
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_resetting",
    "coefficient_of_variation",
    "mfpt_with_resetting",
    "optimal_resetting_rate",
]


def coefficient_of_variation(passage_samples: torch.Tensor) -> torch.Tensor:
    r"""Coefficient of variation :math:`\sqrt{\mathrm{Var}(\tau)}/\mathbb{E}[\tau]`.

    Evans-Majumdar theorem 1: a strictly-accelerating resetting rate
    :math:`r^\star>0` exists iff :math:`\mathrm{CoV}>1`.
    """
    if passage_samples.numel() == 0:
        raise ValueError("passage_samples must have at least one element.")
    mean = passage_samples.mean()
    std = passage_samples.std(unbiased=False)
    return std / mean.clamp_min(1e-12)


def mfpt_with_resetting(passage_samples: torch.Tensor, rate: float | torch.Tensor) -> torch.Tensor:
    r"""Mean first-passage time :math:`\tau_r=(1-\tilde P(r))/(r\tilde P(r))`.

    Args:
        passage_samples: 1-D tensor of unreset first-passage times :math:`\tau_0^{(m)}`.
            Empirical Laplace transform is
            :math:`\hat{\tilde P}(r)=(1/M)\sum_m e^{-r\tau_0^{(m)}}`.
        rate: Resetting rate :math:`r\ge 0` (scalar or tensor). At :math:`r=0`
            the returned value is :math:`\mathbb{E}[\tau_0]` by L'Hôpital's rule.

    Returns:
        Scalar tensor :math:`\tau_r`.
    """
    if passage_samples.ndim != 1:
        raise ValueError(f"passage_samples must be 1-D; got {tuple(passage_samples.shape)}.")
    r = torch.as_tensor(rate, dtype=passage_samples.dtype, device=passage_samples.device)
    if (r < 0).any():
        raise ValueError("rate must be non-negative.")
    if r.abs().max().item() < 1e-12:
        return passage_samples.mean()
    laplace = torch.exp(-r * passage_samples).mean()
    return (1.0 - laplace) / (r * laplace.clamp_min(1e-12))


def optimal_resetting_rate(
    passage_samples: torch.Tensor,
    *,
    r_min: float = 1e-4,
    r_max: float = 10.0,
    n_grid: int = 256,
) -> tuple[float, float]:
    r"""Find :math:`r^\star=\arg\min_r\tau_r` via grid search on :math:`[r_{\min},r_{\max}]`.

    For coefficient of variation > 1 the function :math:`r\mapsto\tau_r` is
    strictly convex on its support and the minimum is unique
    (Evans-Majumdar-Schehr 2020, §3.2). For CoV ≤ 1 the function is monotone
    and :math:`r^\star=r_{\min}` is returned with a flag value.

    Returns:
        ``(r_star, tau_at_r_star)``.
    """
    if r_min <= 0 or r_max <= r_min:
        raise ValueError(f"must have 0 < r_min < r_max; got {r_min, r_max}.")
    grid = torch.logspace(
        torch.log10(torch.tensor(r_min)).item(),
        torch.log10(torch.tensor(r_max)).item(),
        n_grid,
        dtype=passage_samples.dtype,
        device=passage_samples.device,
    )
    taus = torch.stack([mfpt_with_resetting(passage_samples, r) for r in grid.tolist()])
    idx = int(taus.argmin().item())
    return float(grid[idx].item()), float(taus[idx].item())


def apply_resetting(
    x_t: torch.Tensor,
    reset_state: torch.Tensor,
    rate: float,
    dt: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""One reset step:
    :math:`\mathbf{x}_{t+dt}=\mathbf{x}_t-\mathbb{1}[\xi\le r\,dt](\mathbf{x}_t-\mathcal{R}(\hat{\mathbf{y}}))`.

    Args:
        x_t: ``(B, ...)`` current sample.
        reset_state: ``(B, ...)`` data-consistent target
            :math:`\mathcal{R}(\hat{\mathbf{y}})`; broadcast-compatible with ``x_t``.
        rate: Resetting rate :math:`r\ge 0`.
        dt: Time step.
        generator: Optional :class:`torch.Generator` for reproducibility.

    Returns:
        Tensor of the same shape as ``x_t``: positions where reset fired
        have been replaced by ``reset_state``, others unchanged.
    """
    if rate < 0:
        raise ValueError("rate must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    prob_reset = min(rate * dt, 1.0)
    flat_shape = (x_t.shape[0],) + (1,) * (x_t.ndim - 1)
    xi = torch.rand(flat_shape, dtype=x_t.dtype, device=x_t.device, generator=generator)
    reset_mask = xi <= prob_reset
    return torch.where(reset_mask, reset_state.expand_as(x_t), x_t)
