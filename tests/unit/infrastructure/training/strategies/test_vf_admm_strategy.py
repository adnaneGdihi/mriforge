"""``ConcreteVFADMMStrategy``: the OOD acceleration readout seam (VF review 2026-09-03)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator
from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
    ood_acceleration_readout,
    ood_accelerations,
)
from spectramr.infrastructure.training.strategies.vf_admm_strategy import ConcreteVFADMMStrategy


class _RealStackedIdentity(nn.Module):
    """Echoes the real-stacked input, as the ADMM generator's output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_masks: list[bool] = []

    def forward(self, x, marker_prior=None, **kwargs):
        self.seen_masks.append("undersampling_mask" in kwargs or bool(kwargs))
        return x


def _admm(rng):
    class _Probe(ConcreteVFADMMStrategy):
        validation_metrics_computer = None  # the base property needs a validation block

    s = _Probe.__new__(_Probe)
    s.env = SimpleNamespace(generator=_RealStackedIdentity())
    s.device = torch.device("cpu")
    s.simulator = DigitalTwinSimulator(
        im_size=(32, 32),
        enable_motion=False,
        snr_range=(100.0, 100.0),
        enable_undersampling=True,
        acceleration=4.0,
    )
    s.config = SimpleNamespace(
        physics=SimpleNamespace(
            digital_twin=SimpleNamespace(ood_acceleration_range=rng, enable_undersampling=True)
        )
    )
    return s


def test_ood_readout_scores_every_rung_on_the_real_twin_and_restores_it() -> None:
    strat = _admm([16.0])
    target = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    with torch.no_grad():
        scored = strat._score_at_current_twin(target, cache_visuals=True)
        out = ood_acceleration_readout(
            strat.simulator,
            ood_accelerations(strat.config),
            lambda: strat._score_at_current_twin(target, cache_visuals=False),
        )
    assert set(scored) == {"val_psnr"}
    assert strat._last_visual_pred is not None
    assert set(out) == {"val_ood_16x_psnr", "val_ood_accelerations"}
    assert out["val_ood_accelerations"] == 1.0 and torch.isfinite(
        torch.tensor(out["val_ood_16x_psnr"])
    )
    assert strat.simulator.acceleration == 4.0
    assert strat.generator_model.seen_masks == [True, True], (
        "the twin mask reaches the generator on both passes"
    )


def test_no_declared_range_gives_a_zero_count_and_no_extra_pass() -> None:
    strat = _admm(None)
    out = ood_acceleration_readout(
        strat.simulator, ood_accelerations(strat.config), lambda: pytest.fail("no rung, no pass")
    )
    assert out == {"val_ood_accelerations": 0.0}


def test_admm_reads_the_range_and_does_not_claim_the_undersampling_block() -> None:
    assert ConcreteVFADMMStrategy.reads_ood_acceleration_range is True
    assert ConcreteVFADMMStrategy.applies_undersampling is False
    assert "applies_undersampling" not in ConcreteVFADMMStrategy.__dict__
