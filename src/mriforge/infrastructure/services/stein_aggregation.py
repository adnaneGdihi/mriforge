r"""Stein Variational federated posterior aggregation.

Standard federated reconstruction aggregates per-site models by simple
parameter averaging (FedAvg + DP noise [Sheller 2020, Abadi 2016]),
which collapses multi-modal posteriors. **Stein Variational Gradient
Descent (SVGD)** [Liu & Wang 2016, NeurIPS] is a particle-based
sampler that preserves multi-modality:

.. math::

    \mathbf{x}_i \leftarrow \mathbf{x}_i + \tfrac{\epsilon}{n}
        \sum_{j=1}^{n}\!\Big[k(\mathbf{x}_j,\mathbf{x}_i)\,\nabla_{\mathbf{x}_j}\log p(\mathbf{x}_j)
        + \nabla_{\mathbf{x}_j} k(\mathbf{x}_j,\mathbf{x}_i)\Big].

Federated extension [Kassab-Simchowitz 2023]: each site evaluates its
local log-density gradient :math:`\nabla\log p_k(\mathbf{x}_j\mid y_k)` at
the *current particles* and sends them to the server. The server applies
a single SVGD step against the product-of-experts target

.. math::

    p_{\text{fed}}(\mathbf{x}) \propto \prod_{k=1}^{K} p_k(\mathbf{x}\mid y_k)^{1/K}.

Communication is :math:`K\cdot n\cdot d` floats per round — independent
of model size.

Exports:

* :func:`rbf_kernel` — :math:`k(x,y)=\exp(-\|x-y\|^2/h)` with median-bandwidth
  heuristic.
* :func:`svgd_update_step` — one SVGD particle update (10) given a
  log-density-gradient callable.
* :func:`federated_svgd_step` — aggregate :math:`K` site gradients into
  one SVGD update, with optional DP noise injection.
* :func:`product_of_experts_score` — aggregator
  :math:`\sum_k\nabla\log p_k/K` for the federated target.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "federated_svgd_step",
    "product_of_experts_score",
    "rbf_kernel",
    "svgd_update_step",
]


def rbf_kernel(
    particles: torch.Tensor, bandwidth: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""RBF kernel matrix :math:`K_{ij}=\exp(-\|x_i-x_j\|^2/h)`
    and its particle-gradient
    :math:`\nabla_{x_i}K_{ij}=(2/h)(x_j-x_i)K_{ij}`.

    Args:
        particles: ``(n, d)`` particle batch.
        bandwidth: Optional fixed bandwidth :math:`h`. ``None`` uses the
            median-heuristic :math:`h=\mathrm{med}(\|x_i-x_j\|^2)/\log n`.

    Returns:
        ``(kernel, grad_kernel)``: matrices of shape ``(n, n)`` and
        ``(n, n, d)`` respectively.
    """
    if particles.ndim != 2:
        raise ValueError(f"particles must be (n, d); got {tuple(particles.shape)}.")
    n = particles.shape[0]
    if n < 2:
        raise ValueError("need at least 2 particles for SVGD.")
    diff = particles.unsqueeze(0) - particles.unsqueeze(1)  # (n, n, d)
    sq_dist = (diff * diff).sum(dim=-1)
    if bandwidth is None:
        med = torch.median(sq_dist).item()
        bandwidth = max(med / math.log(n + 1.0), 1e-6)
    kernel = torch.exp(-sq_dist / bandwidth)
    # diff[i, j] = x_j - x_i, so ∇_{x_i} k(x_i, x_j) = (2/h)(x_j - x_i) k = (2/h) diff k.
    # (The previous (-diff) negated this, flipping SVGD's repulsive force into an
    # attractive one that collapsed the particles — the opposite of the intent.)
    grad_kernel = (2.0 / bandwidth) * diff * kernel.unsqueeze(-1)  # ∇_{x_i} k(x_i, x_j)
    return kernel, grad_kernel


def svgd_update_step(
    particles: torch.Tensor,
    score_fn,
    *,
    step_size: float = 0.01,
    bandwidth: float | None = None,
) -> torch.Tensor:
    r"""One SVGD step toward the target whose score is ``score_fn``.

    Args:
        particles: ``(n, d)`` current particle batch.
        score_fn: A callable ``score_fn(particles) -> (n, d)`` returning
            :math:`\nabla\log p` at each particle.
        step_size: Gradient-ascent step :math:`\epsilon`.
        bandwidth: RBF bandwidth.

    Returns:
        ``(n, d)`` updated particles.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    score = score_fn(particles)
    if score.shape != particles.shape:
        raise ValueError("score_fn output shape must equal particles shape.")
    kernel, grad_kernel = rbf_kernel(particles, bandwidth=bandwidth)
    n = particles.shape[0]
    # phi*(x_i) = (1/n) Σ_j [ k(x_j, x_i) ∇log p(x_j) + ∇_{x_j} k(x_j, x_i) ]
    phi = (kernel.unsqueeze(-1) * score.unsqueeze(0)).sum(dim=1) / n
    repulsive = grad_kernel.sum(dim=0) / n
    return particles + step_size * (phi + repulsive)


def product_of_experts_score(scores: list[torch.Tensor]) -> torch.Tensor:
    r"""Aggregate K site scores: :math:`\nabla\log p_{\rm fed}=\sum_k\nabla\log p_k/K`.

    Args:
        scores: List of ``(n, d)`` score tensors from :math:`K` sites.

    Returns:
        ``(n, d)`` aggregated score.
    """
    if len(scores) == 0:
        raise ValueError("scores must be non-empty.")
    base = scores[0]
    for s in scores[1:]:
        if s.shape != base.shape:
            raise ValueError("all per-site scores must have identical shape.")
    return torch.stack(scores).mean(dim=0)


def federated_svgd_step(
    particles: torch.Tensor,
    per_site_scores: list[torch.Tensor],
    *,
    step_size: float = 0.01,
    dp_noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Server-side aggregation + SVGD update.

    Each site has already evaluated its local score at the current particles;
    this function aggregates via :func:`product_of_experts_score`, runs one
    SVGD step, and optionally injects isotropic Gaussian DP noise.

    Args:
        particles: ``(n, d)`` current particle batch.
        per_site_scores: List of :math:`K` tensors of shape ``(n, d)``.
        step_size: SVGD step.
        dp_noise_std: Gaussian DP noise standard deviation per parameter.
        generator: Optional :class:`torch.Generator` for the DP noise.

    Returns:
        ``(n, d)`` updated particles.
    """
    if dp_noise_std < 0:
        raise ValueError("dp_noise_std must be non-negative.")
    aggregated_score = product_of_experts_score(per_site_scores)
    if dp_noise_std > 0:
        noise = (
            torch.randn(
                aggregated_score.shape,
                generator=generator,
                device=aggregated_score.device,
            )
            * dp_noise_std
        )
        aggregated_score = aggregated_score + noise
    return svgd_update_step(
        particles,
        lambda _x: aggregated_score,
        step_size=step_size,
    )
