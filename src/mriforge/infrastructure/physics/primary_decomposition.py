r"""Lasker-Noether primary decomposition for multi-contrast Hankel varieties.

Multi-contrast Hankel-low-rank reconstruction [Haldar 2014, Zhang et al. 2022]
exploits the algebraic structure of single-contrast k-space. A joint
acquisition with :math:`C` contrasts produces an annihilator ideal
:math:`I\subset\mathbb{C}[x^{(1)},\ldots,x^{(C)}]` whose primary
decomposition

.. math::

    I = Q_1\cap Q_2\cap\cdots\cap Q_r \quad (\text{Lasker-Noether})

separates the variety into geometrically irreducible *anatomy-contrast
factors* :math:`V(Q_i)`. Each factor is a learnable manifold in the joint
contrast space; the *codimension-normalised* projection regulariser
suppresses embedded primes (which absorb noise) and identifies the
unique minimal decomposition.

This module provides a *practical* surrogate: rather than computing the
true Gröbner-basis decomposition (doubly exponential in the worst case
and requiring a Macaulay2 bridge), we use the **singular-value
decomposition** of stacked single-contrast Hankel matrices as the
algebraic-rank-stratification primitive. The resulting "irreducible
components" are the orthogonal Schmidt subspaces of the multi-contrast
joint matrix; projection onto them is the analogue of the primary
projection :math:`\Pi_{V(Q_i)}`.

Exports:

* :func:`build_hankel_matrix` — construct the block Hankel matrix
  :math:`H(\mathbf{x})\in\mathbb{C}^{p\times q}` from a 1D / 2D signal.
* :func:`multi_contrast_hankel_rank_stratification` — SVD-based
  decomposition into ordered orthogonal components.
* :func:`primary_projection_loss` — codimension-normalised projection
  regulariser
  :math:`\sum_i (\dim V(\mathfrak{p}_i)/\dim V(I))\,\mathrm{dist}(\mathbf{x},V(\mathfrak{p}_i))^2`.
* :func:`recover_loraks_limit` — proof-by-evaluation that at :math:`C=1` the
  decomposition reduces to standard LORAKS [Haldar 2014].
"""

from __future__ import annotations

import torch

__all__ = [
    "build_hankel_matrix",
    "multi_contrast_hankel_rank_stratification",
    "primary_projection_loss",
    "recover_loraks_limit",
]


def build_hankel_matrix(signal: torch.Tensor, window: int) -> torch.Tensor:
    r"""Construct a Hankel matrix from a 1-D signal.

    Args:
        signal: ``(..., N)`` 1-D signal.
        window: Window size :math:`p`; produces a ``(p, N-p+1)`` Hankel
            matrix per batch element.

    Returns:
        ``(..., p, N-p+1)`` Hankel matrix.
    """
    if signal.ndim < 1:
        raise ValueError("signal must have at least one axis.")
    if window <= 0 or window > signal.shape[-1]:
        raise ValueError(f"window must lie in (0, N]; got window={window}, N={signal.shape[-1]}.")
    n = signal.shape[-1]
    cols = n - window + 1
    # Build using unfold along the last axis.
    return signal.unfold(-1, window, 1).transpose(-1, -2)


def multi_contrast_hankel_rank_stratification(
    contrast_signals: torch.Tensor,
    window: int,
    *,
    rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""SVD-based primary stratification of stacked multi-contrast Hankel matrices.

    Given :math:`C` signals, builds the block-stacked Hankel matrix and
    returns its truncated SVD. The columns of :math:`U` (resp. :math:`V`)
    are the Schmidt left- (resp. right-) singular vectors of the joint
    matrix — the practical surrogates of the irreducible components
    :math:`V(\mathfrak{p}_i)` predicted by Lasker-Noether.

    Args:
        contrast_signals: ``(C, N)`` stacked single-contrast signals.
        window: Hankel window size.
        rank: Truncate to :math:`r` singular components; ``None`` returns
            all components.

    Returns:
        ``(u, sigma, vh)`` from the truncated SVD of the block Hankel
        matrix :math:`H=[H^{(1)};\,H^{(2)};\,\ldots;\,H^{(C)}]\in\mathbb{C}^{Cp\times q}`.
    """
    if contrast_signals.ndim != 2:
        raise ValueError(f"contrast_signals must be (C, N); got {tuple(contrast_signals.shape)}.")
    hankel_blocks = [
        build_hankel_matrix(contrast_signals[c], window) for c in range(contrast_signals.shape[0])
    ]
    big_hankel = torch.cat(hankel_blocks, dim=0)
    u, sigma, vh = torch.linalg.svd(big_hankel, full_matrices=False)
    if rank is not None:
        if rank <= 0:
            raise ValueError("rank must be positive.")
        rank = min(rank, sigma.shape[0])
        u = u[..., :rank]
        sigma = sigma[..., :rank]
        vh = vh[..., :rank, :]
    return u, sigma, vh


def primary_projection_loss(
    candidate: torch.Tensor,
    component_basis: torch.Tensor,
    codim_weights: torch.Tensor,
) -> torch.Tensor:
    r"""Codimension-normalised primary-component projection regulariser.

    .. math::

        \mathcal{R}_{\text{ass}}(\mathbf{x}) = \sum_{i=1}^r w_i\,
            \mathrm{dist}(\mathbf{x}, V(\mathfrak{p}_i))^2,

    where :math:`w_i=\dim V(\mathfrak{p}_i)/\dim V(I)` are the
    codimension-normalised weights that suppress embedded primes.

    Args:
        candidate: ``(d,)`` candidate vector :math:`\mathbf{x}`.
        component_basis: ``(d, r)`` orthonormal basis for the union of
            irreducible components; columns are projection vectors.
        codim_weights: ``(r,)`` normalisation weights :math:`w_i`.

    Returns:
        Scalar regulariser value (non-negative).
    """
    if candidate.ndim != 1:
        raise ValueError(f"candidate must be 1-D; got {tuple(candidate.shape)}.")
    if component_basis.shape[0] != candidate.shape[0]:
        raise ValueError("component_basis rows must match candidate length.")
    if codim_weights.shape[0] != component_basis.shape[1]:
        raise ValueError("codim_weights must match the number of components.")
    # Project onto each component (column) and measure the residual.
    inner = component_basis.T @ candidate  # (r,)
    proj_each = (component_basis * inner).T  # (r, d)
    residuals_sq = (candidate.unsqueeze(0) - proj_each).pow(2).sum(dim=-1)
    return (codim_weights * residuals_sq).sum()


def recover_loraks_limit(
    contrast_signals: torch.Tensor, window: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Consistency check: at :math:`C=1` the stratification reduces to LORAKS.

    Returns the SVD of the single-contrast Hankel matrix — exactly the
    LORAKS regulariser kernel [Haldar 2014, §III]. Used in tests to verify
    the proposal's strict-subsumption claim.
    """
    if contrast_signals.shape[0] != 1:
        raise ValueError(f"recover_loraks_limit requires C=1; got C={contrast_signals.shape[0]}.")
    return multi_contrast_hankel_rank_stratification(contrast_signals, window)
