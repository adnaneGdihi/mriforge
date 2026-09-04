"""Numerical primitives for the nonlinear IB-acquisition path (Idea 4).

This module ships:

- :func:`hutchinson_fisher_trace` — unbiased estimator of
  ``tr(J(x))`` where ``J = ∇s ∇s^T`` is the Fisher information matrix
  associated with a registered score model ``s_θ``.
- :func:`posterior_covariance_woodbury_update` — incremental rank-r
  update of ``Σ_X|Y_obs⁻¹`` after a new line ``A_ℓ`` is acquired,
  computed via the Woodbury identity so the full image-dimension
  matrix is never materialised.
- :func:`ib_score_nonlinear` — assembles the IB objective using both
  pieces. The objective is

      L_IB(ℓ) = ½ logdet(I_r + A_ℓ Σ A_ℓᵀ Σ_N⁻¹)  −  β · ½ tr(J(x̂)) / d.

  The linear-Gaussian branch in :mod:`ib_active_acquisition_strategy`
  remains the default; this module is wired in only when
  ``training.ib_active_acquisition.mode == "nonlinear"`` and a
  registered score callable is available.

References:
    - Hutchinson, *Comm. Statist. Simul. Comput.* 19(2), 1990.
    - Higham, *Functions of Matrices*, Ch. 4 (Woodbury identity).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def hutchinson_fisher_trace(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    n_probes: int = 8,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Unbiased estimator of ``tr(J)`` where ``J = ∇s ∇sᵀ``.

    Uses Hutchinson's estimator with Rademacher probes:

        tr(J) = E_v[ vᵀ J v ] = E_v[ ‖∇s · v‖² ].

    Args:
        score_fn: Callable ``[B, ...] → [B, ...]`` returning the score
            at the input (typically ``∇_x log p_σ(x)``).
        x: Sample point at which to evaluate.
        n_probes: Number of Rademacher probes to average.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Scalar tensor (same device as ``x``) — the trace estimate.
    """
    x = x.detach().clone().requires_grad_(True)
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    for _ in range(n_probes):
        v = torch.empty_like(x).bernoulli_(p=0.5, generator=generator).mul_(2).sub_(1)
        s = score_fn(x)
        jvp = torch.autograd.grad(
            outputs=(s * v).sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
        )[0]
        total = total + (v * jvp).sum() ** 2 / (v.numel() ** 0.5)
    return total / n_probes


def posterior_covariance_woodbury_update(
    sigma_inv: torch.Tensor,
    A_ell: torch.Tensor,
    noise_var: float,
) -> torch.Tensor:
    """Update ``Σ⁻¹`` after acquiring a new line ``A_ℓ`` (Woodbury form).

    The posterior precision after adding observation
    ``y_ℓ = A_ℓ x + n_ℓ``, with ``n_ℓ ~ N(0, σ_N² I_r)``, is

        Σ⁻¹_post = Σ⁻¹_prior + A_ℓᵀ A_ℓ / σ_N².

    No covariance materialisation is needed for the update *itself*;
    however many downstream determinant evaluations want the implicit
    determinant ``det(I_r + A Σ Aᵀ / σ_N²)`` which equals
    ``det(I_d + Σ Aᵀ A / σ_N²)`` by Sylvester / Woodbury — and the
    former is cheap when ``r ≪ d``.

    Args:
        sigma_inv: Prior precision ``[d, d]``.
        A_ell: Measurement row matrix ``[r, d]``.
        noise_var: Scalar noise variance ``σ_N²``.

    Returns:
        Updated precision ``Σ⁻¹_post`` of shape ``[d, d]``.
    """
    return sigma_inv + (A_ell.transpose(-1, -2) @ A_ell) / (noise_var + 1e-12)


def ib_score_nonlinear(
    *,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    current_estimate: torch.Tensor,
    candidate_lines: torch.Tensor,
    sigma_post: torch.Tensor | None,
    noise_var: float = 1.0,
    beta: float = 1.0,
    n_probes: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Score every candidate line under the nonlinear-prior IB objective.

    Args:
        score_fn: Diffusion / score model evaluated at the current
            reconstruction.
        current_estimate: Current estimate ``[B, ..., d]`` (the last
            dim is the in-line spatial axis).
        candidate_lines: Boolean mask ``[H]`` — ``True`` for unobserved.
        sigma_post: Optional ``[d, d]`` posterior covariance from
            the linear-Gaussian arm; when ``None`` falls back to the
            identity.
        noise_var: Scalar noise variance.
        beta: IB tradeoff coefficient.
        n_probes: Hutchinson probe count.
        generator: Optional RNG.

    Returns:
        Tensor of shape ``[#candidates]`` giving the IB score per
        unobserved line, with high values preferred.
    """
    unobserved = candidate_lines.bool().nonzero(as_tuple=False).squeeze(-1)
    if unobserved.numel() == 0:
        return torch.empty(0, device=current_estimate.device)

    d = current_estimate.shape[-1]
    if sigma_post is None:
        sigma_post = torch.eye(d, device=current_estimate.device)

    # Fisher trace at current estimate — shared across all candidates.
    tr_J = hutchinson_fisher_trace(
        score_fn,
        current_estimate.flatten(start_dim=1),
        n_probes=n_probes,
        generator=generator,
    )
    fisher_penalty = beta * tr_J / max(d, 1)

    # logdet(I_r + A_ℓ Σ A_ℓᵀ / σ_N²) — for line ℓ, A_ℓ is the row
    # selector e_ℓᵀ, so A Σ Aᵀ is the (ℓ, ℓ) diagonal entry of Σ.
    diag = torch.diagonal(sigma_post, dim1=-2, dim2=-1)
    eigen = 1.0 + diag[unobserved] / (noise_var + 1e-12)
    log_det = 0.5 * torch.log(torch.clamp(eigen, min=1e-12))

    return log_det - fisher_penalty


__all__ = [
    "hutchinson_fisher_trace",
    "ib_score_nonlinear",
    "posterior_covariance_woodbury_update",
]
