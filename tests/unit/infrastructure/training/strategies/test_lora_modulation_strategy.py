"""Tests for LoRAModulationStrategy (B-3.6)."""

from __future__ import annotations

import types

import pytest
import torch

from mriforge.infrastructure.training.strategies.lora_modulation_strategy import (
    LoRAModulationStrategy,
    compute_lora_modulation_loss,
)
from mriforge.models.generators.lora_modulation_net import LoRAModulationNet


def _net() -> LoRAModulationNet:
    return LoRAModulationNet(width=24, n_blocks=2, lora_rank=4)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([7.0, 3.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_lora_modulation_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


@pytest.mark.parametrize("lora_rank", [0, 4])
def test_loss_reduces(lora_rank: int) -> None:
    # Both arms must be non-degenerate translators: the rank=0 control's "still translates"
    # claim is what the rank=4-vs-rank=0 comparison rests on, so it gets its own coverage.
    torch.manual_seed(0)
    m = LoRAModulationNet(width=24, n_blocks=2, lora_rank=lora_rank)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        out = compute_lora_modulation_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_field_sensitivity_monitor_separates_arms() -> None:
    # Mechanism-fires monitor (#16): >0 for an active field-conditioned net, ~0 for BOTH the
    # rank=0 control and the param-matched freeze_field control (the field-blind floor).
    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 16)

    def _sens(net: LoRAModulationNet) -> float:
        strat = object.__new__(LoRAModulationStrategy)
        strat.env = types.SimpleNamespace(generator=net.eval())
        return strat._score_field_sensitivity(x, {"input": x})["val_field_sensitivity"]

    def _activate(net: LoRAModulationNet) -> None:
        with torch.no_grad():
            for c in [net.lift, *net.block_convs, net.head]:
                c.lora_up.weight.normal_(0.0, 0.3)
                c.alpha_mlp.weight.normal_(0.0, 0.5)

    active = LoRAModulationNet(width=24, n_blocks=2, lora_rank=4)
    _activate(active)
    assert _sens(active) > 1e-4  # mechanism fires

    assert _sens(LoRAModulationNet(width=24, n_blocks=2, lora_rank=0)) < 1e-6  # no adapter

    frozen = LoRAModulationNet(width=24, n_blocks=2, lora_rank=4, freeze_field=True)
    _activate(frozen)
    assert _sens(frozen) < 1e-6  # capacity present but field-blind


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(LoRAModulationStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._lm_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(LoRAModulationStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._lm_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range_uses_both_fields() -> None:
    m = _net().eval()
    strat = object.__new__(LoRAModulationStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"field_strength": torch.tensor([0.1, 1.5]), "field_strength_target": torch.tensor([7.0, 3.0])},
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_both_fields() -> None:
    import pytest

    strat = object.__new__(LoRAModulationStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(
            torch.rand(1, 1, 16, 16), {"field_strength": torch.tensor([0.1])}  # missing target
        )


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "lora_modulation" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "lora_modulation" in TrainingStrategyConfigSchema.model_fields
