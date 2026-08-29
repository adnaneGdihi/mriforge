"""Tests for the mri_perfusion regime losses."""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.signal_models.perfusion_kinetics import (
    extended_tofts_forward,
    parker_population_aif,
)
from mriforge.models.losses.perfusion_losses import (
    AIFConsistencyLoss,
    PerfusionMapSmoothnessLoss,
    PerfusionPhysiologicalBoxLoss,
    ToftsResidualLoss,
)
from mriforge.models.losses.registry import LossRegistry


def test_perfusion_losses_registered_and_tagged() -> None:
    from mriforge.config.schemas.enums import Regime

    meta = LossRegistry._loss_domains
    for name in (
        "tofts_residual",
        "aif_consistency",
        "perfusion_physiological_box",
        "perfusion_map_smoothness",
    ):
        assert name in meta
        assert Regime.PERFUSION in (meta[name].get("workflows") or frozenset())


def test_tofts_residual_zero_at_true_curve() -> None:
    t = torch.linspace(0, 300, 151)
    aif = parker_population_aif(t)
    ktrans, ve, vp = torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.02)
    curve = extended_tofts_forward(t, aif, ktrans, ve, vp)
    loss = ToftsResidualLoss()
    assert loss(t, aif, ktrans, ve, vp, curve).item() < 1e-5


def test_physiological_box_penalises_violations() -> None:
    loss = PerfusionPhysiologicalBoxLoss()
    ok = loss(torch.tensor(0.1), torch.tensor(0.3), torch.tensor(0.2))
    assert ok.item() == 0.0
    # ve + vp > 1 and negative Ktrans both penalised.
    bad = loss(torch.tensor(-0.1), torch.tensor(0.8), torch.tensor(0.5))
    assert bad.item() > 0.0


def test_aif_consistency_zero_on_match() -> None:
    aif = parker_population_aif(torch.linspace(0, 60, 61))
    assert AIFConsistencyLoss()(aif, aif).item() == 0.0


def test_map_smoothness_zero_on_constant() -> None:
    flat = torch.ones(1, 1, 8, 8)
    assert PerfusionMapSmoothnessLoss()(flat).item() == 0.0
    noisy = torch.randn(1, 1, 8, 8)
    assert PerfusionMapSmoothnessLoss()(noisy).item() > 0.0
