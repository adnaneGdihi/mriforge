"""Tests for MonotoneFieldStrategy (B-2.6)."""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.monotone_field_strategy import (
    MonotoneFieldStrategy,
    compute_monotone_field_loss,
)
from mriforge.models.generators.monotone_field_generator import (
    MonotoneFieldOrderedGenerator,
)


def _net() -> MonotoneFieldOrderedGenerator:
    return MonotoneFieldOrderedGenerator(width=24, n_bases=4)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_monotone_field_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the inline objective via the
    loss-SSOT seam; the inline l1 placeholder is skipped (no double-count)."""
    from mriforge.data.batch_types import BatchAdapter
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss
    from mriforge.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(MonotoneFieldStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))
    strat._mf_lambda_l1 = 1.0
    strat._mf_lambda_monotone = 0.0

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_monotone_penalty_wired_when_enabled() -> None:
    out = compute_monotone_field_loss(_net(), _batch(), lambda_monotone=0.5)
    assert "loss_monotone" in out
    assert torch.isfinite(out["loss_total"])


def test_monotone_penalty_ceiling_tracks_model_field_range() -> None:
    # Review #4: the b_hi ceiling must derive from the model's trained field range
    # (10**tau_max), not a hardcoded 7.0. A model whose tau_max=2 (field up to 100T)
    # must still produce a finite, non-degenerate penalty for a high target field.
    m = MonotoneFieldOrderedGenerator(width=24, n_bases=4, tau_min=-1.0, tau_max=2.0)
    batch = dict(_batch())
    batch["field_strength_target"] = torch.tensor([50.0, 80.0])  # > old 7.0 ceiling
    out = compute_monotone_field_loss(m, batch, lambda_monotone=0.5)
    assert torch.isfinite(out["loss_monotone"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = compute_monotone_field_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_validation_forward_renders_at_field() -> None:
    m = _net().eval()
    strat = object.__new__(MonotoneFieldStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(MonotoneFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "monotone_field" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "monotone_field" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(MonotoneFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])
