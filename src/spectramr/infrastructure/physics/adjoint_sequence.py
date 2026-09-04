r"""Adjoint sequence optimisation for marker-anchored MRI sequence design.

Implements ``synthetic_marker_research_plan.md`` §4.6 (Method VI): a
differentiable forward Bloch simulator plus an outer optimisation loop
that uses synthetic markers as gradient-stabilisation anchors.

Forward model
-------------
For a tissue with ``T1``, ``T2``, ``M0`` driven by a sequence of pulses
``φ = (α_n, TR_n, TE_n, ...)`` the discrete Bloch evolution is:

.. math::

    \dot{\mathbf m} = F(\mathbf m, \phi(t)),
    \quad \mathbf m \in \mathbb R^3 \text{ (Mx, My, Mz)}.

The stationary signal after ``N`` repetitions of an SE-ish RF–TE–TR
block is approximated by the Ernst formula
``S = M0 (1 − e^(−TR/T1)) e^(−TE/T2)`` for an unsaturated 90° pulse.
For the optimisation we track the autograd graph through the sequence
of ``α``, ``TR``, ``TE`` parameters so that ``dS/dφ`` is exact.

Adjoint (Pontryagin) method
---------------------------
For an inner-loop loss ``L`` on the reconstructed image ``x̂(φ)``, the
gradient ``dL/dφ`` is computed via the adjoint state ``λ(t)`` which
satisfies the backward ODE

.. math::
    \dot \lambda = -\Big(\frac{\partial F}{\partial \mathbf m}\Big)^\top \lambda,
    \quad \lambda(T) = \nabla_{\mathbf m} L,

so that

.. math::
    \frac{dL}{d\phi} = \int_0^T \lambda(t)^\top
                       \frac{\partial F}{\partial \phi}\,dt.

For the small parameter set in MRI sequence design (``α``, ``TR``,
``TE``), torch's reverse-mode autograd implements this adjoint
implicitly when the forward Bloch model is written as a torch
computation graph, which is what we do here.

Marker-anchored rank augmentation (Spec §4.6 theorem)
-----------------------------------------------------
Define ``J_φ = E[G(r) G(r)^T]`` with ``G(r) = ∇_φ x̂(r)``. Adding
markers strictly increases ``rank(J_φ)`` if marker Jacobians are
linearly independent of anatomy Jacobians on ``Ω_a``. The
:func:`marker_anchored_outer_loss` helper combines an anatomy-side
reconstruction loss with a marker-anchor term that explicitly forces
the gradient flow through the marker pixels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Differentiable forward Bloch simulator (steady-state approximation)
# ──────────────────────────────────────────────────────────────────────


def steady_state_se_signal(
    t1: torch.Tensor,
    t2: torch.Tensor,
    m0: torch.Tensor,
    *,
    tr: torch.Tensor,
    te: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Differentiable steady-state spin-echo (SE) signal.

    .. math::
        S = M_0\,\sin\alpha\,
            \frac{1 - e^{-TR/T_1}}{1 - \cos\alpha\,e^{-TR/T_1}}\,
            e^{-TE/T_2}

    With ``alpha=π/2`` (default) this collapses to the standard SE
    signal :math:`S = M_0 (1 - e^{-TR/T_1}) e^{-TE/T_2}`.

    All inputs broadcast.
    """
    if alpha is None:
        alpha = torch.tensor(torch.pi / 2.0, device=t1.device)
    e1 = torch.exp(-tr / t1.clamp(min=1e-6))
    e2 = torch.exp(-te / t2.clamp(min=1e-6))
    sin_a = torch.sin(alpha)
    cos_a = torch.cos(alpha)
    return m0 * sin_a * (1.0 - e1) / (1.0 - cos_a * e1).clamp(min=1e-12) * e2


def steady_state_ir_se_signal(
    t1: torch.Tensor,
    t2: torch.Tensor,
    m0: torch.Tensor,
    *,
    tr: torch.Tensor,
    te: torch.Tensor,
    ti: torch.Tensor,
) -> torch.Tensor:
    r"""Differentiable steady-state IR-SE (inversion-recovery SE) signal.

    .. math::
        S = M_0\,\bigl(1 - 2 e^{-TI/T_1} + e^{-TR/T_1}\bigr)\,
            e^{-TE/T_2}
    """
    e_ti = torch.exp(-ti / t1.clamp(min=1e-6))
    e_tr = torch.exp(-tr / t1.clamp(min=1e-6))
    e2 = torch.exp(-te / t2.clamp(min=1e-6))
    return m0 * (1.0 - 2.0 * e_ti + e_tr) * e2


# ──────────────────────────────────────────────────────────────────────
# Marker-anchored optimisation
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SequenceOptResult:
    """Output of :func:`optimise_sequence_adjoint`."""

    phi_hat: dict[str, torch.Tensor]  # final sequence parameters
    loss_history: list[float]
    grad_norm_history: list[float]
    best_loss: float
    best_phi: dict[str, torch.Tensor]


def optimise_sequence_adjoint(
    forward_signal_fn: Callable[..., torch.Tensor],
    tissue_t1: torch.Tensor,
    tissue_t2: torch.Tensor,
    tissue_m0: torch.Tensor,
    target_signal: torch.Tensor,
    initial_phi: dict[str, torch.Tensor],
    *,
    marker_t1: torch.Tensor | None = None,
    marker_t2: torch.Tensor | None = None,
    marker_m0: torch.Tensor | None = None,
    target_marker_signal: torch.Tensor | None = None,
    lambda_marker: float = 1.0,
    n_iters: int = 200,
    lr: float = 5e-2,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> SequenceOptResult:
    r"""Outer optimisation of sequence parameters via reverse-mode adjoint.

    Loss:

    .. math::
        \mathcal L(\phi) = \|S(\phi; T_1^a, T_2^a, M_0^a) - S^*\|^2
                          + \lambda_m \|S(\phi; T_1^m, T_2^m, M_0^m) - S^{m,*}\|^2

    The second term is the marker-anchor — its gradient stabilises the
    optimisation against ill-conditioning of the anatomy-only Jacobian
    (Method VI rank-augmentation theorem).

    Args:
        forward_signal_fn: Callable e.g. :func:`steady_state_se_signal`
                           or :func:`steady_state_ir_se_signal`.
        tissue_t1, tissue_t2, tissue_m0: anatomy tissue parameters.
        target_signal:     desired anatomy signal (e.g. CNR-optimal).
        initial_phi:       dict mapping ``'tr'``, ``'te'``, ``'ti'``,
                           ``'alpha'`` → initial-value scalars or
                           1-D tensors.
        marker_t1, ...:    marker tissue parameters (None disables the
                           marker term).
        target_marker_signal: desired marker signal.
        lambda_marker:     weight on the marker-anchor term.
        n_iters:           Adam iterations.
        lr:                Adam learning rate.
        bounds:            Optional per-parameter ``(min, max)`` clamps
                           applied at every step.

    Returns:
        :class:`SequenceOptResult`.
    """
    # Make φ trainable
    phi: dict[str, torch.Tensor] = {
        k: v.detach().clone().requires_grad_(True) for k, v in initial_phi.items()
    }
    optim = torch.optim.Adam(list(phi.values()), lr=lr)
    loss_hist: list[float] = []
    grad_hist: list[float] = []
    best_loss = float("inf")
    best_phi: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in phi.items()}

    has_marker = (
        marker_t1 is not None
        and marker_t2 is not None
        and marker_m0 is not None
        and target_marker_signal is not None
    )

    for _ in range(n_iters):
        optim.zero_grad()
        s_anat = forward_signal_fn(tissue_t1, tissue_t2, tissue_m0, **phi)
        loss = ((s_anat - target_signal) ** 2).mean()
        if has_marker:
            s_mark = forward_signal_fn(marker_t1, marker_t2, marker_m0, **phi)
            loss = loss + lambda_marker * ((s_mark - target_marker_signal) ** 2).mean()
        loss.backward()
        # Clip grads for stability
        torch.nn.utils.clip_grad_norm_(list(phi.values()), max_norm=10.0)
        # Track the gradient norm summed over all parameters
        gn = 0.0
        for v in phi.values():
            if v.grad is not None:
                gn += float(v.grad.norm().item()) ** 2
        grad_hist.append(gn**0.5)
        optim.step()
        # Apply bounds
        if bounds:
            with torch.no_grad():
                for k, (lo, hi) in bounds.items():
                    if k in phi:
                        phi[k].clamp_(lo, hi)
        l = float(loss.item())
        loss_hist.append(l)
        if l < best_loss:
            best_loss = l
            best_phi = {k: v.detach().clone() for k, v in phi.items()}

    return SequenceOptResult(
        phi_hat={k: v.detach() for k, v in phi.items()},
        loss_history=loss_hist,
        grad_norm_history=grad_hist,
        best_loss=best_loss,
        best_phi=best_phi,
    )


def jacobian_rank_augmentation_test(
    anatomy_jacobians: torch.Tensor,
    marker_jacobians: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[int, int]:
    r"""Test the Spec §4.6 rank-augmentation theorem on a problem.

    Returns ``(rank_anatomy_only, rank_with_markers)``. By the spec
    theorem, the second value is ``≥`` the first, with strict
    inequality iff the marker Jacobians span a direction outside the
    anatomy-Jacobian column space.

    Args:
        anatomy_jacobians: ``(N_a, p)`` Jacobian rows for anatomy voxels.
        marker_jacobians:  ``(N_m, p)`` Jacobian rows for marker voxels.
        eps: numerical tolerance for SVD-rank.
    """
    rank_a = int(torch.linalg.matrix_rank(anatomy_jacobians, tol=eps).item())
    combined = torch.cat([anatomy_jacobians, marker_jacobians], dim=0)
    rank_combined = int(torch.linalg.matrix_rank(combined, tol=eps).item())
    return rank_a, rank_combined


__all__ = [
    "SequenceOptResult",
    "jacobian_rank_augmentation_test",
    "optimise_sequence_adjoint",
    "steady_state_ir_se_signal",
    "steady_state_se_signal",
]
