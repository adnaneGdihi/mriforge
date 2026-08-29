"""Tests for FieldFNOStrategy (B-1.6/3.7)."""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.field_fno_strategy import (
    FieldFNOStrategy,
    compute_field_fno_loss,
)
from mriforge.models.generators.field_conditioned_fno import FieldConditionedFNO


def _net() -> FieldConditionedFNO:
    return FieldConditionedFNO(width=16, modes=8, n_layers=2, field_hidden=16)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_field_fno_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_field_fno_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_validation_forward_renders_at_field() -> None:
    m = _net().eval()
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def _perturb_field_mlp(m: FieldConditionedFNO) -> FieldConditionedFNO:
    """Push the zero-init field-token MLP off identity — what training does."""
    torch.manual_seed(1)
    with torch.no_grad():
        for spec in m.spectral:
            last = spec.field_mlp[-1]
            last.weight.add_(torch.randn_like(last.weight) * 0.2)
            last.bias.add_(torch.randn_like(last.bias) * 0.2)
    return m


def test_score_fno_zero_when_mechanism_inert() -> None:
    # field_token_enable=False -> the field branch is skipped -> identical render at any
    # field -> witness EXACTLY 0. This is the one-knob control (#16/#17).
    m = FieldConditionedFNO(
        width=16, modes=8, n_layers=2, field_hidden=16, field_token_enable=False
    ).eval()
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    x = torch.rand(2, 1, 16, 16)
    score = strat._score_fno(x, {"input": x})
    assert score["val_fno_field_sensitivity"] == 0.0


def test_score_fno_zero_at_init_rises_when_trained() -> None:
    # The multiplier is zero-init = identity, so a GREEN SMOKE on an UNTRAINED model is
    # field-invariant by construction (witness 0) — the #16 facade trap. Perturbing the
    # field MLP off identity (what training does) makes the witness positive, certifying
    # at run time that the field-conditioning fired.
    torch.manual_seed(0)
    m = _net().eval()  # default field_token_enable=True, zero-init field MLP
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    x = torch.rand(2, 1, 16, 16)
    assert strat._score_fno(x, {"input": x})["val_fno_field_sensitivity"] == 0.0
    _perturb_field_mlp(m)
    assert strat._score_fno(x, {"input": x})["val_fno_field_sensitivity"] > 1e-4


def test_score_fno_self_gates_on_bad_input() -> None:
    # Non-4D / empty input must not crash validation — the witness self-gates to {}.
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    assert strat._score_fno(None, {}) == {}
    assert strat._score_fno(torch.zeros(0), {}) == {}
    assert strat._score_fno(torch.rand(2, 16, 16), {}) == {}  # 3-D, not [B,C,H,W]


def test_validation_step_merges_field_sensitivity_witness(monkeypatch) -> None:
    # The witness must reach the returned metrics dict — the seam the run actually reads.
    # Stub the parent validation_step (resolved by super() via the MRO) to isolate the merge.
    import mriforge.infrastructure.training.strategies.field_fno_strategy as mod

    monkeypatch.setattr(
        mod.ReconstructionTrainingStrategy,
        "validation_step",
        lambda self, *a, **k: {"val_psnr": 30.0},
    )
    m = _perturb_field_mlp(_net().eval())
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    x = torch.rand(2, 1, 16, 16)
    out = FieldFNOStrategy.validation_step(
        strat, x, torch.rand(2, 1, 16, 16), field_strength_target=torch.tensor([7.0, 1.5])
    )
    assert "val_psnr" in out  # parent metrics preserved
    assert out["val_fno_field_sensitivity"] > 1e-4  # witness merged in


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_fno" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "field_fno" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(generator=_net())

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) are folded onto the inline L1 via the
    loss-SSOT seam; the inline l1 placeholder is skipped (no double-count)."""
    from mriforge.data.batch_types import BatchAdapter
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss
    from mriforge.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(FieldFNOStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(),
        losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))
    strat._fno_lambda_l1 = 1.0

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert "loss_l1" in out          # the strategy's own inline l1 component
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out  # popped, not leaked
