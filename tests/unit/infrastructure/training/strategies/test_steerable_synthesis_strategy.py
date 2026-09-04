"""Tests for SteerableSynthesisStrategy (B-1.4)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.steerable_synthesis_strategy import (
    SteerableSynthesisStrategy,
    compute_steerable_synthesis_loss,
)
from spectramr.models.generators.steerable_field_unet import SteerableFieldUNet


def _net() -> SteerableFieldUNet:
    return SteerableFieldUNet(base_width=8, n_blocks=2)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_steerable_synthesis_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_steerable_synthesis_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard): the canonical pipeline forwards a TrainingBatch, not a
    # dict. The guard must accept any mapping exposing .get.
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(SteerableSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._ss_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(SteerableSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._ss_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16),
            target_batch=torch.rand(2, 1, 16, 16),
            epoch=0,
            batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_synthesises_at_source_field() -> None:
    m = _net().eval()
    strat = object.__new__(SteerableSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength=torch.tensor([0.1, 3.0]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_clamps_to_unit_range() -> None:
    # REGRESSION (#20): the generator output is intentionally unbounded (raw, for training
    # gradients), but validation must clamp to [0,1] so SSIM/PSNR (which auto-detect
    # data_range from the [0,1] target) grade in-range predictions and the arm-vs-arm
    # comparison is not biased by out-of-range pixels.
    torch.manual_seed(0)
    m = _net().eval()
    # perturb so the raw output genuinely leaves [0,1]
    with torch.no_grad():
        for p in m.proj.parameters():
            p.mul_(8.0)
    strat = object.__new__(SteerableSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16), {"use_dc": False}, field_strength=torch.tensor([0.1, 3.0])
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(SteerableSynthesisStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "steerable_synthesis" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "steerable_synthesis" in TrainingStrategyConfigSchema.model_fields


# --- Multi-contrast: contrast_id threaded to the equivariant synthesiser ---


def test_contrast_id_threaded_to_model() -> None:
    from spectramr.infrastructure.training.strategies.steerable_synthesis_strategy import (
        compute_steerable_synthesis_loss,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([1, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 3.0]),
        "contrast_id": cid,
    }
    compute_steerable_synthesis_loss(spy, batch)
    assert seen["cid"] is cid
