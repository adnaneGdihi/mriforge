"""Tests for BlochFieldStrategy (B-1.8)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.bloch_field_strategy import (
    BlochFieldStrategy,
    compute_bloch_field_loss,
)
from spectramr.models.generators.bloch_field_bottleneck import BlochFieldBottleneck


def _net() -> BlochFieldBottleneck:
    return BlochFieldBottleneck(width=24)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_bloch_field_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the inline L1 via the loss-SSOT
    seam; the inline l1 placeholder is skipped (no double-count)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(BlochFieldStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))
    strat._bf_lambda_l1 = 1.0

    out = strat._compute_losses_impl(input_batch=_batch(), target_batch=None, epoch=0)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = compute_bloch_field_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_validation_forward_renders_at_field() -> None:
    m = _net().eval()
    strat = object.__new__(BlochFieldStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([7.0, 7.0]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(BlochFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort-wide): the canonical pipeline converts the loader dict to a
    # TrainingBatch (BatchAdapter.from_dict) and calls _compute_losses_impl with
    # input_batch=<tensor> and batch=<TrainingBatch>. The original guard
    # `if not isinstance(batch, dict): raise` rejected the TrainingBatch, crashing ALL
    # MICCAI arms at step 0. The earlier tests only fed raw dicts, so this slipped
    # through. The guard must accept any mapping that exposes .get (dict OR TrainingBatch).
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(BlochFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._bf_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(BlochFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._bf_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16),
            target_batch=torch.rand(2, 1, 16, 16),
            epoch=0,
            batch=torch.rand(2, 1, 16, 16),
        )


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "bloch_field" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "bloch_field" in TrainingStrategyConfigSchema.model_fields
