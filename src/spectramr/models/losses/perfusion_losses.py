"""Perfusion kinetics losses (regime: ``mri_perfusion``).

Regime-specific losses that (with the kinetic signal model and the perfusion
metrics) move ``mri_perfusion`` off EVAL_ONLY. Each is tagged
``workflows={Regime.PERFUSION}`` so the maturity ledger sees real loss coverage.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.signal_models.perfusion_kinetics import (
    extended_tofts_forward,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="tofts_residual",
    domain="image",
    workflows=frozenset({Regime.PERFUSION}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class ToftsResidualLoss(nn.Module):
    """Self-supervised fit residual: does the forward Tofts model reproduce the curve?

    Given predicted ``(Ktrans, ve, vp)`` maps, the measured AIF and the measured
    tissue concentration curve, penalise the mismatch between the forward-model
    prediction and the observed curve.

    Args:
        forward_model: the ``(t_s, aif, ktrans, ve, vp) -> curve`` map. Defaults
            to :func:`extended_tofts_forward`. ``PerfusionKineticMappingStrategy``
            passes the model it resolved from ``data.perfusion.kinetic_model``
            via the ``SignalModelRegistry``, so that key dispatches for real
            instead of being re-hardcoded here. The default keeps the class
            directly constructible in tests and honours its own name.
    """

    def __init__(
        self,
        forward_model: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model or extended_tofts_forward

    def forward(
        self,
        t_s: torch.Tensor,
        aif: torch.Tensor,
        ktrans: torch.Tensor,
        ve: torch.Tensor,
        vp: torch.Tensor,
        measured_curve: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.forward_model(t_s, aif, ktrans, ve, vp)
        return (predicted - measured_curve).abs().mean()


@register_loss(
    name="aif_consistency",
    domain="image",
    workflows=frozenset({Regime.PERFUSION}),
)
class AIFConsistencyLoss(nn.Module):
    """Penalise a learned AIF deviating from a reference (population) AIF."""

    def forward(self, aif_pred: torch.Tensor, aif_reference: torch.Tensor) -> torch.Tensor:
        return (aif_pred - aif_reference).abs().mean()


@register_loss(
    name="perfusion_physiological_box",
    domain="image",
    workflows=frozenset({Regime.PERFUSION}),
)
class PerfusionPhysiologicalBoxLoss(nn.Module):
    """Soft box constraints on kinetic maps: ``Ktrans, ve, vp ≥ 0`` and ``ve + vp ≤ 1``."""

    def forward(self, ktrans: torch.Tensor, ve: torch.Tensor, vp: torch.Tensor) -> torch.Tensor:
        neg = torch.clamp(-ktrans, min=0.0) + torch.clamp(-ve, min=0.0) + torch.clamp(-vp, min=0.0)
        overflow = torch.clamp((ve + vp) - 1.0, min=0.0)
        return neg.mean() + overflow.mean()


@register_loss(
    name="perfusion_map_smoothness",
    domain="image",
    workflows=frozenset({Regime.PERFUSION}),
)
class PerfusionMapSmoothnessLoss(nn.Module):
    """Total-variation smoothness prior on a parameter map ``[B, C, H, W]``."""

    def forward(self, param_map: torch.Tensor) -> torch.Tensor:
        dh = (param_map[..., 1:, :] - param_map[..., :-1, :]).abs().mean()
        dw = (param_map[..., :, 1:] - param_map[..., :, :-1]).abs().mean()
        return dh + dw


__all__ = [
    "AIFConsistencyLoss",
    "PerfusionMapSmoothnessLoss",
    "PerfusionPhysiologicalBoxLoss",
    "ToftsResidualLoss",
]
