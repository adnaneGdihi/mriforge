"""Tests for McCannFieldPathStrategy (B-3.9)."""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.mccann_field_path_strategy import (
    McCannFieldPathStrategy,
    compute_mccann_loss,
)
from mriforge.models.generators.mccann_geodesic_icnn import McCannGeodesicICNN


def _net() -> McCannGeodesicICNN:
    return McCannGeodesicICNN(hidden_channels=16, n_layers=3)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([7.0, 3.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_mccann_loss(_net(), _batch())
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
        out = compute_mccann_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(McCannFieldPathStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._mc_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(McCannFieldPathStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._mc_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_clamps_and_uses_both_fields() -> None:
    m = _net().eval()
    with torch.no_grad():
        m.icnn.map_scale_raw.fill_(3.0)
        for p in m.icnn.z_raw:
            p.normal_(0.0, 2.0)
    strat = object.__new__(McCannFieldPathStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"field_strength": torch.tensor([0.1, 1.5]), "field_strength_target": torch.tensor([7.0, 3.0])},
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_both_fields() -> None:
    import pytest

    strat = object.__new__(McCannFieldPathStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(
            torch.rand(1, 1, 16, 16), {"field_strength": torch.tensor([0.1])}  # missing target
        )


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "mccann_field_path" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "mccann_field_path" in TrainingStrategyConfigSchema.model_fields
