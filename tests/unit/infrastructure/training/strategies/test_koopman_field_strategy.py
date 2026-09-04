"""Tests for KoopmanFieldStrategy (B-3.10)."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.koopman_field_strategy import (
    KoopmanFieldStrategy,
    compute_koopman_field_loss,
)
from spectramr.models.generators.koopman_field_propagator import KoopmanFieldPropagator


def _net(**kw) -> KoopmanFieldPropagator:
    return KoopmanFieldPropagator(width=24, koopman_dim=16, n_blocks=2, **kw)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 1.5]),
        "field_strength_target": torch.tensor([7.0, 3.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_koopman_field_loss(_net(), _batch())
    assert {"loss_total", "loss_l1", "loss_recon"} <= set(out)
    assert torch.isfinite(out["loss_total"])


@pytest.mark.parametrize("use_koopman_semigroup", [True, False])
def test_loss_reduces(use_koopman_semigroup: bool) -> None:
    # Both arms must be non-degenerate translators (the comparison rests on it).
    torch.manual_seed(0)
    m = _net(use_koopman_semigroup=use_koopman_semigroup)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        out = compute_koopman_field_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._kf_lambda_l1 = 1.0
    strat._kf_lambda_recon = 0.5
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._kf_lambda_l1 = 1.0
    strat._kf_lambda_recon = 0.5
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16),
            target_batch=torch.rand(2, 1, 16, 16),
            epoch=0,
            batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range_uses_both_fields() -> None:
    m = _net().eval()
    strat = object.__new__(KoopmanFieldStrategy)
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
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(
            torch.rand(1, 1, 16, 16),
            {"field_strength": torch.tensor([0.1])},  # missing target
        )


def test_score_koopman_emits_monitors() -> None:
    # semigroup arm: spectral abscissa (after A!=0) + field & input sensitivity; nonlinear:
    # no abscissa but both sensitivities. logging_service None -> no warning crash.
    x = torch.rand(2, 1, 16, 16)

    sg = _net(use_koopman_semigroup=True)
    with torch.no_grad():
        sg.generator.normal_(0.0, 0.3)
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=sg.eval())
    strat.logging_service = None
    out = strat._score_koopman(x, {"input": x})
    assert "val_koopman_spectral_abscissa" in out
    assert out["val_field_sensitivity"] > 1e-4  # field mechanism fires once A != 0
    assert out["val_input_sensitivity"] > 1e-4  # output depends on the input (not collapsed)

    nl = _net(use_koopman_semigroup=False)
    strat2 = object.__new__(KoopmanFieldStrategy)
    strat2.env = types.SimpleNamespace(generator=nl.eval())
    strat2.logging_service = None
    out2 = strat2._score_koopman(x, {"input": x})
    assert "val_koopman_spectral_abscissa" not in out2  # no linear operator
    assert "val_field_sensitivity" in out2
    assert out2["val_input_sensitivity"] > 1e-4  # the bare autoencoder is input-dependent


def test_input_sensitivity_warns_on_collapsed_encoder() -> None:
    # An encoder that maps everything to a constant -> measurement-independent output (#20):
    # val_input_sensitivity ~0 must fire a warning.
    x = torch.rand(2, 1, 16, 16)
    m = _net(use_koopman_semigroup=True).eval()
    with torch.no_grad():  # force the encoder to a constant (collapsed)
        for p in m.encoder.parameters():
            p.zero_()
    warnings: list[str] = []
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat.logging_service = types.SimpleNamespace(log_warning=lambda msg: warnings.append(msg))
    out = strat._score_koopman(x, {"input": x})
    assert out["val_input_sensitivity"] < 1e-3
    assert any("input_sensitivity" in w for w in warnings)


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the inline objective via the
    loss-SSOT seam; the inline l1 placeholder is skipped (no double-count)."""
    from spectramr.data.batch_types import BatchAdapter
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()}
    )
    strat.config = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
            kspace_losses=[],
            complex_losses=[],
        )
    )
    strat._kf_lambda_l1 = 1.0
    strat._kf_lambda_recon = 0.5
    strat._kf_lambda_stability = 0.0

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_stability_penalty_wired_and_differentiable() -> None:
    # Semigroup arm with a positive-definite symmetric part -> mu2(A)=1>0 -> relu fires.
    # The numerical abscissa is differentiable, so grad reaches the operator A.
    m = _net(use_koopman_semigroup=True)
    with torch.no_grad():
        m.generator.copy_(torch.eye(m.koopman_dim))  # symmetric part = I -> mu2 = 1
    out = compute_koopman_field_loss(m, _batch(), lambda_stability=0.1)
    assert "loss_stability" in out
    assert float(out["loss_stability"].detach()) > 0.0
    assert torch.isfinite(out["loss_total"])
    out["loss_total"].backward()
    assert m.generator.grad is not None and torch.isfinite(m.generator.grad).all()


def test_stability_penalty_skipped_on_nonlinear_arm() -> None:
    # The nonlinear ablation carries no linear operator A (model.generator is None), so the
    # stability term is structurally skipped even with lambda_stability>0 (one-knob #17).
    m = _net(use_koopman_semigroup=False)
    assert m.generator is None
    out = compute_koopman_field_loss(m, _batch(), lambda_stability=0.1)
    assert "loss_stability" not in out
    assert torch.isfinite(out["loss_total"])


def test_lambda_stability_config_read_and_stamped() -> None:
    # The knob must be read from training.koopman_field.lambda_stability (#15).
    logs: list[str] = []
    strat = object.__new__(KoopmanFieldStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat.logging_service = types.SimpleNamespace(
        log_info=lambda m: logs.append(m), log_warning=lambda m: logs.append(m)
    )
    strat._log_config_features = lambda *_a, **_k: None
    cfg = types.SimpleNamespace(
        koopman_field=types.SimpleNamespace(lambda_recon=0.5, lambda_stability=0.05)
    )
    # The inline L1 weight is read from the loss-weight table (losses.image_losses), not
    # from the training block (mrixfields review 2026-09-03).
    strat.config = types.SimpleNamespace(
        training=cfg,
        losses=types.SimpleNamespace(
            image_losses=[{"name": "l1", "weight": 1.0}], kspace_losses=[], complex_losses=[]
        ),
    )
    strat._verify_strategy_config = lambda **_k: None
    strat._setup_strategy_specific_components()
    assert strat._kf_lambda_stability == 0.05
    assert any("lambda_stability=0.05" in m for m in logs)


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "koopman_field" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "koopman_field" in TrainingStrategyConfigSchema.model_fields
