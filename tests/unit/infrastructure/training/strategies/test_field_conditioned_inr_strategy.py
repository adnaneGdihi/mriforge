"""Tests for FieldConditionedINRStrategy + FieldConditionedINR (B-2.8)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.field_conditioned_inr_strategy import (
    FieldConditionedINRStrategy,
    _render,
    compute_field_conditioned_inr_loss,
)
from spectramr.models.generators.field_conditioned_inr import FieldConditionedINR


def _net() -> FieldConditionedINR:
    return FieldConditionedINR(feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_builder_image_losses_folded_via_seam() -> None:
    """Regime-2 suppression: declarative tv/log_spectral fold onto the inline L1 via the
    loss-SSOT seam; the inline l1 placeholder is skipped (no double-count)."""
    from spectramr.models.losses.spectral_loss import LogSpectralLoss
    from spectramr.models.losses.standard_losses import TotalVariationLoss

    strat = object.__new__(FieldConditionedINRStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(),
        losses={"l1": TotalVariationLoss(), "tv": TotalVariationLoss(),
                "log_spectral": LogSpectralLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "tv", "weight": 0.05},
                      {"name": "log_spectral", "weight": 0.1}],
        kspace_losses=[], complex_losses=[]))
    strat._inr_use_field = True
    strat._inr_lambda_l1 = 1.0

    out = strat._compute_losses_impl(input_batch=_batch(), target_batch=None, epoch=0)
    assert "loss_builder_aux" in out and "loss_tv" in out and "loss_log_spectral" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_rejects_complex() -> None:
    import pytest

    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_loss_keys_and_finite() -> None:
    out = compute_field_conditioned_inr_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = compute_field_conditioned_inr_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_field_is_load_bearing_after_perturbation() -> None:
    # ANTI-FACADE (#16): once FiLM/field_mlp move off identity init, the continuous
    # field must change the render (the field is genuine conditioning, not inert).
    m = _net()
    with torch.no_grad():
        for fl in m.siren.film_layers:
            if fl is not None:
                for p in fl.parameters():
                    p.normal_(std=0.3)
        for p in m.field_mlp.parameters():
            p.normal_(std=0.3)
    x = torch.rand(2, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo, hi)


def test_ablation_is_field_agnostic() -> None:
    # The one-knob ablation: with use_field_conditioning=False, _render passes a
    # constant zero field, so the output is INVARIANT to the requested field — even
    # with a perturbed (non-identity) network. This proves the knob genuinely gates it.
    m = _net()
    with torch.no_grad():
        for fl in m.siren.film_layers:
            if fl is not None:
                for p in fl.parameters():
                    p.normal_(std=0.3)
        for p in m.field_mlp.parameters():
            p.normal_(std=0.3)
    m.eval()
    x = torch.rand(1, 1, 16, 16)
    a = _render(m, x, torch.tensor([0.1]), use_field_conditioning=False)
    b = _render(m, x, torch.tensor([7.0]), use_field_conditioning=False)
    assert torch.allclose(a, b)  # field-agnostic when conditioning is off


def _perturb(m):
    with torch.no_grad():
        for fl in m.siren.film_layers:
            if fl is not None:
                for p in fl.parameters():
                    p.normal_(std=0.3)
        for p in m.field_mlp.parameters():
            p.normal_(std=0.3)
    return m


def test_loss_path_respects_field_conditioning_knob() -> None:
    # The TRAINING loss path must honor use_field_conditioning. With a perturbed net,
    # the loss must change with the target field when ON, and be field-invariant when
    # OFF (the ablation truly removes field coupling at training, not just validation).
    m = _perturb(_net())
    x, y = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
    hi = {"input": x, "target": y, "field_strength_target": torch.tensor([7.0, 7.0])}
    lo = {"input": x, "target": y, "field_strength_target": torch.tensor([0.1, 0.1])}
    on_hi = compute_field_conditioned_inr_loss(m, hi, use_field_conditioning=True)["loss_total"]
    on_lo = compute_field_conditioned_inr_loss(m, lo, use_field_conditioning=True)["loss_total"]
    assert not torch.allclose(on_hi, on_lo)  # field ON -> field changes the loss
    off_hi = compute_field_conditioned_inr_loss(m, hi, use_field_conditioning=False)["loss_total"]
    off_lo = compute_field_conditioned_inr_loss(m, lo, use_field_conditioning=False)["loss_total"]
    assert torch.allclose(off_hi, off_lo)  # field OFF -> loss invariant to field


def test_validation_forward_renders_at_field() -> None:
    m = _net().eval()
    strat = object.__new__(FieldConditionedINRStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._inr_use_field = True
    pred = strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert pred.shape == (2, 1, 16, 16)


def test_validation_forward_raises_without_field() -> None:
    import pytest

    m = _net()
    strat = object.__new__(FieldConditionedINRStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._inr_use_field = True
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


def test_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.models.registry import MODEL_REGISTRY

    assert "field_conditioned_inr" in MODEL_REGISTRY
    assert "field_conditioned_inr" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "field_conditioned_inr" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldConditionedINRStrategy)
    strat.env = types.SimpleNamespace(generator=_net())

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the field-INR ---


def test_contrast_id_threaded_to_model() -> None:
    from spectramr.infrastructure.training.strategies.field_conditioned_inr_strategy import (
        compute_field_conditioned_inr_loss,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([1, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_field_conditioned_inr_loss(spy, batch)
    assert seen["cid"] is cid
