r"""Concomitant-phase residual loss (CAUR / idea 1 §1.2).

Penalises the magnitude of the *residual* concomitant-phase
correction emitted by an analytical-plus-residual head:

.. math::

    \mathcal{L}_{c} = \lambda \, \lVert \Delta\phi_\theta \rVert_2^2

where :math:`\Delta\phi_\theta` is the residual correction the network
adds on top of the closed-form Bernstein 1998 prediction, and
:math:`\lambda` is the weight. The residual head is *only* used when
``physics.concomitant.model = "analytical_plus_residual"`` — for the
default ``analytical`` mode this loss is omitted from the recipe.

Why penalise the residual?
--------------------------
The Bernstein closed-form already captures the leading concomitant
term. The residual head is a fall-back for un-modelled effects (gradient
imperfections, eddy currents, etc.). Allowing it to absorb arbitrary
phase content would let the network "explain away" data inconsistencies
that should be handled by the data-consistency layer instead. A small
ridge penalty keeps the head honest.

Reference
---------
Plan §1.2 of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


@register_loss(
    name="concomitant_phase_residual",
    aliases=["ConcomitantPhaseResidualLoss"],
    domain="kspace",
    # The loss penalizes a real-valued residual-phase head output (radians),
    # not k-space data itself, so it is domain-agnostic and is routinely
    # declared under losses.image_losses by image-domain reconstruction arms.
    compatible_with=["image"],
)
class ConcomitantPhaseResidualLoss(nn.Module):
    r"""Squared-L2 ridge on the concomitant-phase residual head output.

    Args:
        weight: ``λ`` multiplier. Must be ≥ 0.
        reduction: ``"mean"`` (default) → ``λ · mean(|Δφ|²)``;
            ``"sum"`` → ``λ · sum(|Δφ|²)``.
    """

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        self.weight = float(weight)
        self.reduction = reduction

    def forward(
        self,
        residual_phase: torch.Tensor,
        target: torch.Tensor | None = None,  # unused; kept for API parity
        **_: object,
    ) -> torch.Tensor:
        r"""Compute :math:`\lambda \, \lVert \Delta\phi \rVert_2^2`.

        Args:
            residual_phase: Real-valued residual-phase tensor (radians).
            target: Ignored. Present so the loss matches the standard
                ``forward(prediction, target)`` signature used by the
                builder layer.

        Returns:
            Scalar tensor.
        """
        sq = residual_phase.pow(2)
        if self.reduction == "mean":
            penalty = sq.mean()
        else:
            penalty = sq.sum()
        return self.weight * penalty


__all__ = ["ConcomitantPhaseResidualLoss"]
