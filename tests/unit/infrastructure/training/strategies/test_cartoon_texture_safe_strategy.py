"""Tests for CartoonTextureSafeStrategy (B-2.5)."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from mriforge.infrastructure.training.strategies.cartoon_texture_safe_strategy import (
    CartoonTextureSafeStrategy,
    compute_cartoon_texture_loss,
)
from mriforge.models.losses.cartoon_texture_loss import CartoonTextureSafeLoss


class _TinyRestorer(nn.Module):
    """A minimal image->image restorer stand-in (B-2.5 is generator-agnostic)."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.body(x)


def _batch() -> dict:
    return {"input": torch.rand(2, 1, 32, 32), "target": torch.rand(2, 1, 32, 32)}


def test_loss_keys_and_finite() -> None:
    out = compute_cartoon_texture_loss(_TinyRestorer(), _batch(), CartoonTextureSafeLoss(n_iter=5))
    assert {"loss_total", "loss_l1", "loss_cartoon", "loss_texture"} <= set(out)
    assert torch.isfinite(out["loss_total"])


@pytest.mark.parametrize("lambda_ct", [0.0, 0.5])
def test_loss_reduces(lambda_ct: float) -> None:
    torch.manual_seed(0)
    m = _TinyRestorer()
    ct = CartoonTextureSafeLoss(n_iter=5)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = compute_cartoon_texture_loss(m, batch, ct, lambda_ct=lambda_ct)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(CartoonTextureSafeStrategy)
    strat.env = types.SimpleNamespace(generator=_TinyRestorer())
    strat._ct_lambda_l1 = 1.0
    strat._ct_lambda_ct = 0.5
    strat._ct_loss = CartoonTextureSafeLoss(n_iter=5)
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    strat = object.__new__(CartoonTextureSafeStrategy)
    strat.env = types.SimpleNamespace(generator=_TinyRestorer())
    strat._ct_lambda_l1 = 1.0
    strat._ct_lambda_ct = 0.5
    strat._ct_loss = CartoonTextureSafeLoss(n_iter=5)
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range() -> None:
    m = _TinyRestorer().eval()
    strat = object.__new__(CartoonTextureSafeStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._ct_loss = CartoonTextureSafeLoss(n_iter=5)
    pred = strat._validation_forward(torch.rand(2, 1, 32, 32), {}).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_score_cartoon_emits_fidelity() -> None:
    torch.manual_seed(0)
    m = _TinyRestorer().eval()
    strat = object.__new__(CartoonTextureSafeStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._ct_loss = CartoonTextureSafeLoss(n_iter=5)
    out = strat._score_cartoon(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32), {})
    assert "val_cartoon_fidelity" in out
    assert 0.0 <= out["val_cartoon_fidelity"] <= 1.0


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "cartoon_texture_safe" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "cartoon_texture_safe" in TrainingStrategyConfigSchema.model_fields
