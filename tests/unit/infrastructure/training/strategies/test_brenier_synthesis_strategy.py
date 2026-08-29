"""Tests for BrenierSynthesisStrategy (B-1.5)."""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.brenier_synthesis_strategy import (
    BrenierSynthesisStrategy,
    compute_brenier_loss,
)
from mriforge.models.generators.brenier_icnn import BrenierICNN


def _net() -> BrenierICNN:
    return BrenierICNN(hidden_channels=16, n_layers=3)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_brenier_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_brenier_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(BrenierSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._br_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(BrenierSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._br_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_clamps_to_unit_range() -> None:
    m = _net().eval()
    with torch.no_grad():  # blow up the map so the raw output leaves [0,1]
        m.icnn.map_scale_raw.fill_(3.0)
        for p in m.icnn.z_raw:
            p.normal_(0.0, 2.0)
    strat = object.__new__(BrenierSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16), {"use_dc": False}, field_strength=torch.tensor([0.1, 3.0])
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(BrenierSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "brenier_synthesis" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "brenier_synthesis" in TrainingStrategyConfigSchema.model_fields


# --- Multi-contrast: contrast_id threaded to the Brenier map ---


def test_contrast_id_threaded_to_model() -> None:
    from mriforge.infrastructure.training.strategies.brenier_synthesis_strategy import (
        compute_brenier_loss,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 3.0]),
        "contrast_id": cid,
    }
    compute_brenier_loss(spy, batch)
    assert seen["cid"] is cid
