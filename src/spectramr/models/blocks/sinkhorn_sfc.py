"""Sinkhorn-relaxed metric space-filling-curve linearizer.

The hard MST + DFS path in
:class:`spectramr.models.blocks.metric_sfc.MetricSFCLinearizer` produces
a fixed permutation at ``__init__`` time. That works for inference
and a fixed-template training regime but breaks autograd: the
permutation itself isn't trainable.

This module supplies the *Sinkhorn-relaxed* variant called for in
plan §4.1: the hard permutation matrix is replaced by a doubly-
stochastic matrix obtained from the entropic optimal-transport
projection

.. math::

   P_\\tau \\;=\\; \\mathrm{Sinkhorn}\\!\\left(\\exp(-C / \\tau)\\right),

where :math:`C_{ij} = \\| x_i - x_j \\|_g` is the metric-graph
distance (or any user-supplied cost), and :math:`\\tau > 0` is the
entropic regularizer. As :math:`\\tau \\to 0` the matrix converges
to the hard assignment; as :math:`\\tau \\to \\infty` it converges
to the uniform marginal-product matrix.

In the GeoMamba-ULF forward pass the soft permutation acts on
tokens as a single matrix multiply:

.. math::

   y_t \\;=\\; \\sum_s P_{ts}\\, x_s,

which is differentiable end-to-end through the Sinkhorn iterations
(see Mena et al. 2018, "Learning Latent Permutations with
Gumbel-Sinkhorn Networks"; Cuturi 2013, "Sinkhorn Distances:
Lightspeed Computation of Optimal Transport").

This module is small and standalone — the strategy / generator can
plug it in as a drop-in alternative to the hard linearizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = 20,
    tau: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Sinkhorn-Knopp normalisation in log-space.

    Args:
        log_alpha: ``(*, N, N)`` log-cost matrix (negative cost,
            i.e. log of the unnormalised affinities).
        n_iters: number of row/column normalisation iterations.
        tau: entropic temperature. Higher τ → more uniform; lower
            τ → closer to a hard permutation. Typical: 0.1–1.0.

    Returns:
        ``(*, N, N)`` doubly-stochastic matrix.
    """
    if log_alpha.shape[-1] != log_alpha.shape[-2]:
        raise ValueError(
            f"log_alpha must be square in the last two dims; got {tuple(log_alpha.shape)}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0; got {tau}")
    log_alpha = log_alpha / tau
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


class SinkhornSFCLinearizer(nn.Module):
    """Soft, differentiable foreground-restricted SFC permutation.

    Treats the foreground voxel positions as a 1D point cloud, with
    pairwise costs initialized from the same metric used by
    :class:`MetricSFCLinearizer`. The cost matrix is a learnable
    parameter so the network can refine the traversal during
    training; Sinkhorn projection gives a doubly-stochastic
    permutation at every step.

    Args:
        n_foreground: number of foreground voxels (sequence length).
        init_cost: optional ``(N, N)`` initial cost matrix. Use the
            metric-graph distance from
            :func:`MetricSFCLinearizer._build_metric_graph` for a
            warm-start; defaults to a learned identity-modulated
            matrix when ``None``.
        sinkhorn_iters: number of Sinkhorn iterations per forward.
        tau: entropic temperature (lower = sharper permutation).
        learnable_tau: when True, ``tau`` is a learnable parameter
            so the network can anneal sharpness on its own.
    """

    def __init__(
        self,
        n_foreground: int,
        init_cost: torch.Tensor | None = None,
        sinkhorn_iters: int = 20,
        tau: float = 1.0,
        learnable_tau: bool = False,
    ) -> None:
        super().__init__()
        if n_foreground <= 0:
            raise ValueError(f"n_foreground must be positive; got {n_foreground}")
        self.n_foreground = int(n_foreground)
        self.sinkhorn_iters = int(sinkhorn_iters)

        if init_cost is None:
            cost = torch.zeros(n_foreground, n_foreground)
        else:
            if init_cost.shape != (n_foreground, n_foreground):
                raise ValueError(
                    f"init_cost must be ({n_foreground}, {n_foreground}); "
                    f"got {tuple(init_cost.shape)}"
                )
            cost = init_cost.detach().clone().float()
        # Negate so that small distances → large affinities → likely matches.
        self.cost = nn.Parameter(-cost)

        if learnable_tau:
            self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
            self._tau_const: float | None = None
        else:
            self.register_buffer("_tau_buf", torch.tensor(float(tau)))
            self.log_tau = None  # type: ignore[assignment]
            self._tau_const = float(tau)

    @property
    def tau(self) -> torch.Tensor | float:
        if self.log_tau is not None:
            return self.log_tau.exp()
        return self._tau_const if self._tau_const is not None else float(self._tau_buf.item())

    def soft_permutation(self) -> torch.Tensor:
        """Compute the current doubly-stochastic permutation matrix."""
        tau_value: torch.Tensor | float
        if self.log_tau is not None:
            # Keep tau as a tensor (do NOT detach) so a loss on the permutation
            # backprops into ``log_tau`` — ``sinkhorn`` divides by tau in a
            # differentiable way. Detaching to a float made learnable_tau inert
            # (pitfall #16).
            tau_value = self.log_tau.exp()
        elif self._tau_const is not None:
            tau_value = self._tau_const
        else:
            tau_value = float(self._tau_buf.item())
        return sinkhorn(self.cost, n_iters=self.sinkhorn_iters, tau=tau_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the soft permutation to a token sequence.

        Args:
            x: ``(B, N, d_model)`` token sequence already restricted
                to the foreground (caller is responsible for
                gathering only foreground voxels — typically by
                composing this layer with a precomputed mask).

        Returns:
            ``(B, N, d_model)`` permuted sequence.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B, N, d_model); got {tuple(x.shape)}")
        if x.shape[1] != self.n_foreground:
            raise ValueError(f"Expected sequence length {self.n_foreground}; got {x.shape[1]}")
        P = self.soft_permutation()  # (N, N)
        # (B, N, d_model) ⊗ (N, N) → einsum 'bnd, nm -> bmd'.
        return torch.einsum("bnd,nm->bmd", x, P)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the transpose of the soft permutation to undo `forward`.

        Note: a doubly-stochastic matrix is not exactly invertible
        unless it is a permutation. ``reverse`` therefore returns
        ``y @ P^T`` which is the right-inverse only in expectation
        as ``tau → 0``. For low-temperature (sharp) regimes the
        approximation is tight enough for visualisation /
        round-trip checks; for hard reconstruction the strategy
        should anneal ``tau`` and snap to a permutation matrix
        (see ``hard_permutation``).
        """
        if y.ndim != 3:
            raise ValueError(f"Expected (B, N, d_model); got {tuple(y.shape)}")
        P = self.soft_permutation()
        return torch.einsum("bnd,mn->bmd", y, P)  # transpose via index swap

    def hard_permutation(self) -> torch.Tensor:
        """Snap the soft permutation to the nearest hard permutation.

        Used at inference time to recover an exact permutation
        matrix from the doubly-stochastic relaxation. Implemented
        as a Hungarian assignment on the negative log-affinities;
        falls back to argmax-per-row if scipy is unavailable.
        """
        with torch.no_grad():
            cost_np = (-self.cost).detach().cpu().numpy()
            try:
                from scipy.optimize import linear_sum_assignment

                row_ind, col_ind = linear_sum_assignment(cost_np)
            except ImportError:  # pragma: no cover - scipy is a hard dep already
                col_ind = cost_np.argmax(axis=1)
                row_ind = list(range(len(col_ind)))
            P = torch.zeros_like(self.cost)
            for r, c in zip(row_ind, col_ind, strict=True):
                P[int(r), int(c)] = 1.0
        return P


__all__ = ["SinkhornSFCLinearizer", "sinkhorn"]
