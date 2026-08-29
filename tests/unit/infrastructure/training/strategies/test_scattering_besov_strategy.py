"""Tests for ScatteringBesovStrategy (B-1.3)."""

from __future__ import annotations

import types

import pytest
import torch

from mriforge.infrastructure.training.strategies.scattering_besov_strategy import (
    ScatteringBesovStrategy,
    compute_scattering_besov_loss,
)
from mriforge.models.generators.scattering_field_translator import ScatteringFieldTranslator
from mriforge.models.losses.scattering_besov_loss import ScatteringBesovLoss


def _net() -> ScatteringFieldTranslator:
    return ScatteringFieldTranslator(width=24, n_blocks=2)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([0.1, 1.5]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_scattering_besov_loss(_net(), _batch(), ScatteringBesovLoss())
    assert {"loss_total", "loss_l1", "loss_scatter", "loss_besov"} <= set(out)
    assert torch.isfinite(out["loss_total"])


@pytest.mark.parametrize("lambda_scatter", [0.0, 0.5])
def test_loss_reduces(lambda_scatter: float) -> None:
    # Both arms (with and without the detail prior) must be non-degenerate trainable translators.
    torch.manual_seed(0)
    m = _net()
    sl = ScatteringBesovLoss()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = compute_scattering_besov_loss(m, batch, sl, lambda_scatter=lambda_scatter)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._sb_lambda_l1 = 1.0
    strat._sb_lambda_scatter = 0.5
    strat._scatter_loss = ScatteringBesovLoss()
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._sb_lambda_l1 = 1.0
    strat._sb_lambda_scatter = 0.5
    strat._scatter_loss = ScatteringBesovLoss()
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_compute_losses_requires_field_strength() -> None:
    # Missing source field -> a clear ValueError, not a cryptic KeyError (pitfall #9). The
    # Tier-2 probe injects field_strength, so this real-path guard is what protects training.
    batch = {"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}
    with pytest.raises(ValueError, match="field_strength"):
        compute_scattering_besov_loss(_net(), batch, ScatteringBesovLoss())


def test_validation_forward_in_unit_range() -> None:
    m = _net().eval()
    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._scatter_loss = ScatteringBesovLoss()
    pred = strat._validation_forward(
        torch.rand(2, 1, 32, 32), {"field_strength": torch.tensor([0.1, 1.5])}
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_detail_monitor_emits_ratio() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._scatter_loss = ScatteringBesovLoss()
    x = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    out = strat._score_detail(x, target, {"input": x, "field_strength": torch.tensor([0.1, 1.5])})
    assert "val_detail_energy_ratio" in out and "val_detail_energy_target" in out
    assert out["val_detail_energy_ratio"] >= 0.0


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "scattering_besov" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "scattering_besov" in TrainingStrategyConfigSchema.model_fields


def _seam_config() -> types.SimpleNamespace:
    """A minimal config exposing declarative image-loss weights for the seam."""
    return types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[
                {"name": "charbonnier", "weight": 0.3},
                {"name": "hfen", "weight": 0.2},
            ],
            kspace_losses=[],
            complex_losses=[],
        )
    )


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses on env.losses are folded onto loss_total (SSOT seam)."""
    from mriforge.data.batch_types import BatchAdapter
    from mriforge.models.losses.charbonnier_loss import CharbonnierLoss
    from mriforge.models.losses.hfen_loss import HFENLoss

    torch.manual_seed(0)
    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(),
        losses={"charbonnier": CharbonnierLoss(), "hfen": HFENLoss()},
    )
    strat.config = _seam_config()
    strat._sb_lambda_l1 = 1.0
    strat._sb_lambda_scatter = 0.5
    strat._scatter_loss = ScatteringBesovLoss()

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    # The builder terms fired and were folded, with per-name components recorded.
    assert "loss_builder_aux" in out
    assert "loss_charbonnier" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    # Grad-carrying tensors are popped, never leaked into the returned loss dict.
    assert "prediction" not in out and "target_image" not in out


def test_seam_is_noop_without_env_losses() -> None:
    """No env.losses -> the seam adds nothing (inline-only behaviour preserved)."""
    from mriforge.data.batch_types import BatchAdapter

    strat = object.__new__(ScatteringBesovStrategy)
    strat.env = types.SimpleNamespace(generator=_net())  # no `losses` attribute
    strat._sb_lambda_l1 = 1.0
    strat._sb_lambda_scatter = 0.5
    strat._scatter_loss = ScatteringBesovLoss()

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert "loss_builder_aux" not in out
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the translator ---


def test_contrast_id_threaded_to_model() -> None:
    from mriforge.infrastructure.training.strategies.scattering_besov_strategy import (
        compute_scattering_besov_loss,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    class _ScatterSpy:
        def compute(self, pred, y):
            z = torch.tensor(0.0)
            return {"loss_total": z, "loss_scatter": z, "loss_besov": z}

    cid = torch.tensor([1, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 3.0]),
        "contrast_id": cid,
    }
    compute_scattering_besov_loss(spy, batch, _ScatterSpy())
    assert seen["cid"] is cid
