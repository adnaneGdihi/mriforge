"""Tests for FisherRaoGeodesicStrategy (B-3.4)."""

from __future__ import annotations

import types

import torch

from mriforge.infrastructure.training.strategies.fisher_rao_geodesic_strategy import (
    FisherRaoGeodesicStrategy,
    compute_fisher_rao_loss,
)
from mriforge.models.generators.fisher_rao_geodesic_net import FisherRaoGeodesicNet


def _net() -> FisherRaoGeodesicNet:
    return FisherRaoGeodesicNet(width=24, n_blocks=2)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([7.0, 3.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_fisher_rao_loss(_net(), _batch())
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
        out = compute_fisher_rao_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from mriforge.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FisherRaoGeodesicStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._fr_lambda_l1 = 1.0
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(FisherRaoGeodesicStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._fr_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16),
            target_batch=torch.rand(2, 1, 16, 16),
            epoch=0,
            batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range_uses_both_fields() -> None:
    m = _net().eval()
    with torch.no_grad():
        m.head.weight.normal_(0.0, 2.0)
        m.head.bias.normal_(0.0, 2.0)
    strat = object.__new__(FisherRaoGeodesicStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {
            "field_strength": torch.tensor([0.1, 1.5]),
            "field_strength_target": torch.tensor([7.0, 3.0]),
        },
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_both_fields() -> None:
    import pytest

    strat = object.__new__(FisherRaoGeodesicStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(
            torch.rand(1, 1, 16, 16),
            {"field_strength": torch.tensor([0.1])},  # missing target
        )


def test_strategy_registered_and_config_mounted() -> None:
    from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
    from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "fisher_rao_geodesic" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "fisher_rao_geodesic" in TrainingStrategyConfigSchema.model_fields


# --- Multi-contrast: contrast_id threaded to the geodesic net ---


def test_contrast_id_threaded_to_model() -> None:
    from mriforge.infrastructure.training.strategies.fisher_rao_geodesic_strategy import (
        compute_fisher_rao_loss,
    )

    seen: dict = {}

    def spy(x, *, field_strength, field_strength_target, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 1])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_fisher_rao_loss(spy, batch)
    assert seen["cid"] is cid


def test_effective_l1_weight_is_lambda_l1_not_image_losses() -> None:
    """Regression (b34-M1): the effective L1 weight is the training-block lambda_l1
    threaded as ``compute_fisher_rao_loss(..., lambda_l1=...)``, NOT ``losses.image_losses``.

    FisherRaoGeodesicStrategy is an inline-loss strategy that never routes ``image_losses``
    through the LossBuilder, so ``loss_total`` must scale EXACTLY with ``lambda_l1`` while
    ``loss_l1`` stays weight-independent. This pins the config-facade fix: the YAML
    ``image_losses[l1].weight`` is a schema placeholder, never the gradient authority.
    """
    net = _net()
    batch = _batch()
    one = compute_fisher_rao_loss(net, batch, lambda_l1=1.0)
    ten = compute_fisher_rao_loss(net, batch, lambda_l1=10.0)
    # loss_l1 is the raw reconstruction error, independent of the weight.
    assert torch.allclose(one["loss_l1"], ten["loss_l1"])
    # loss_total == lambda_l1 * loss_l1 exactly (no image_losses contribution).
    assert torch.allclose(one["loss_total"], one["loss_l1"])
    assert torch.allclose(ten["loss_total"], 10.0 * ten["loss_l1"])
