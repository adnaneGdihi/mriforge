"""Tests for FieldFlowStrategy: velocity-regression math + path-composition (B-3.1)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.field_flow_strategy import (
    compute_field_flow_losses,
    integrate_field_flow,
)
from spectramr.models.generators.field_velocity_unet import FieldVelocityUNet
from tests.utils.config_block_stub import block_stub


class _OracleVelocity(nn.Module):
    """Returns a fixed per-unit-tau velocity regardless of x / field (for exact tests)."""

    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_):
        return self.velocity.expand_as(x)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 0.5]),
    }


def test_loss_keys_and_finite() -> None:
    m = FieldVelocityUNet(width=8)
    out = compute_field_flow_losses(m, _batch(), lambda_straightness=0.01)
    assert {"loss_total", "loss_velocity", "loss_straightness"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_velocity_loss_zero_for_matching_oracle() -> None:
    # If the model returns exactly the secant velocity u = (x_t - x_s)/span, the
    # velocity loss must be ~0. Use a single-sample batch so u is a single tensor.
    x_s = torch.zeros(1, 1, 4, 4)
    x_t = torch.ones(1, 1, 4, 4)
    b_s, b_t = 1.0, 10.0  # tau_s=0, tau_t=1, span=1 -> u = x_t - x_s = ones
    batch = {
        "input": x_s,
        "target": x_t,
        "field_strength": torch.tensor([b_s]),
        "field_strength_target": torch.tensor([b_t]),
    }
    oracle = _OracleVelocity(torch.ones(1, 1, 4, 4))  # u = 1 since span=1
    out = compute_field_flow_losses(oracle, batch, lambda_straightness=0.0)
    assert float(out["loss_velocity"]) < 1e-10


def test_integrate_recovers_endpoint_constant_velocity() -> None:
    # dx/dtau = w (constant) integrated over [tau_s, tau_t] = x0 + w*(tau_t - tau_s).
    # b_s=1 (tau=0), b_t=10 (tau=1) -> span 1 -> x0 + w.
    x0 = torch.zeros(1, 1, 4, 4)
    w = torch.full((1, 1, 4, 4), 0.7)
    oracle = _OracleVelocity(w)
    out = integrate_field_flow(oracle, x0, b_s=1.0, b_t=10.0, n_steps=8, solver="euler")
    assert torch.allclose(out, x0 + w, atol=1e-5)


def test_path_composition_semigroup_exact() -> None:
    # Phi_{b_s->b_t} == Phi_{b_m->b_t} o Phi_{b_s->b_m} for a constant velocity field
    # (exact for an oracle, by integrating over tau). b_s=0.1, b_m=1.0, b_t=7.0.
    torch.manual_seed(0)
    x0 = torch.rand(1, 1, 4, 4)
    w = torch.full((1, 1, 4, 4), 0.3)
    oracle = _OracleVelocity(w)
    single = integrate_field_flow(oracle, x0, 0.1, 7.0, n_steps=16, solver="euler")
    mid = integrate_field_flow(oracle, x0, 0.1, 1.0, n_steps=8, solver="euler")
    composed = integrate_field_flow(oracle, mid, 1.0, 7.0, n_steps=8, solver="euler")
    assert torch.allclose(single, composed, atol=1e-5)


def test_velocity_regression_reduces() -> None:
    torch.manual_seed(0)
    m = FieldVelocityUNet(width=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_field_flow_losses(m, batch, lambda_straightness=0.0)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_velocity"].detach())
    assert out is not None and first is not None
    assert float(out["loss_velocity"].detach()) < first


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_flow" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def _bare_strategy(generator) -> object:
    import types

    from spectramr.infrastructure.training.strategies.field_flow_strategy import (
        FieldFlowStrategy,
    )

    strat = object.__new__(FieldFlowStrategy)
    strat.env = types.SimpleNamespace(generator=generator)
    strat._ff_n_steps = 8
    strat._ff_solver = "euler"
    return strat


def test_validation_forward_reads_fields_from_kwargs_not_just_batch_context() -> None:
    # REALISTIC contract: the canonical validation dispatch passes batch_context as
    # {'use_dc': False, 'measured_kspace': None} and forwards the per-sample fields
    # as KWARGS (the pipeline gated seam). _validation_forward must integrate using
    # those — comparing a raw velocity to the target would be a metric mismatch
    # (pitfall #18). (The earlier test hand-fed the keys via batch_context and so
    # masked the KeyError the adversarial review found.)
    w = torch.full((1, 1, 4, 4), 0.3)
    strat = _bare_strategy(_OracleVelocity(w))
    x0 = torch.zeros(1, 1, 4, 4)
    pred = strat._validation_forward(
        x0,
        {"use_dc": False, "measured_kspace": None},  # the real batch_context
        field_strength=torch.tensor([1.0]),
        field_strength_target=torch.tensor([10.0]),
    )
    assert torch.allclose(pred, x0 + w, atol=1e-5)  # tau span = 1 -> x0 + w


def test_validation_forward_raises_clearly_when_fields_absent() -> None:
    # No silent fallback (pitfall #9): a clear error, not a KeyError, when the
    # field endpoints are missing entirely.
    strat = _bare_strategy(_OracleVelocity(torch.zeros(1, 1, 4, 4)))
    with pytest.raises(ValueError, match="field_strength"):
        strat._validation_forward(torch.zeros(1, 1, 4, 4), {"use_dc": False})


def test_validation_step_declares_field_keys_for_pipeline_seam() -> None:
    # The pipeline forwards a field only if validation_step DECLARES it. Guard that
    # the signature carries the keys so the seam (train.py) actually forwards them.
    import inspect

    from spectramr.infrastructure.training.strategies.field_flow_strategy import (
        FieldFlowStrategy,
    )

    params = inspect.signature(FieldFlowStrategy.validation_step).parameters
    assert "field_strength" in params
    assert "field_strength_target" in params


class _CountingOracle(_OracleVelocity):
    """Oracle velocity that counts forward calls (proves integrate_field_flow ran)."""

    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__(velocity)
        self.calls = 0

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **kw):
        self.calls += 1
        return super().forward(x, field_strength=field_strength)


def test_validation_step_end_to_end_no_batch_context_collision() -> None:
    # REGRESSION for the batch_context double-pass: drive the FULL validation_step
    # chain the way train.py:select_validation_extra_fields -> strategy.validation_step
    # does (per-sample fields as KWARGS). The override folds them into batch_context
    # and calls super().validation_step(..., batch_context=...); with the mixin's
    # .get (pre-fix) this raised "multiple values for argument 'batch_context'".
    # The prior tests called _validation_forward directly and MASKED this.
    import types

    from spectramr.infrastructure.training.strategies.field_flow_strategy import (
        FieldFlowStrategy,
    )

    oracle = _CountingOracle(torch.full((2, 1, 4, 4), 0.3))
    strat = object.__new__(FieldFlowStrategy)
    # generator_model is a property -> env.generator; the mixin calls .eval() on it.
    strat.env = types.SimpleNamespace(generator=oracle)
    strat._ff_n_steps = 8
    strat._ff_solver = "euler"
    # metrics off -> exercise the unpack/batch_context/_validation_forward path only
    strat.state = types.SimpleNamespace(
        config=types.SimpleNamespace(
            training=None,
            # folded to `validation.scoring.enable_image_metrics`
            validation=block_stub("validation", compute_image_metrics=False),
        )
    )
    out = strat.validation_step(
        torch.zeros(2, 1, 4, 4),
        torch.zeros(2, 1, 4, 4),
        field_strength=torch.tensor([1.0, 1.0]),
        field_strength_target=torch.tensor([10.0, 10.0]),
    )
    assert isinstance(out, dict)  # returned without TypeError
    assert oracle.calls > 0  # integrate_field_flow actually ran through the chain


def test_velocity_norm_knob_changes_loss() -> None:
    # The velocity_norm knob must be live (the registered FieldFlowVelocityLoss is
    # the one invoked, not a facade) — l1 and l2 give different values.
    m = FieldVelocityUNet(width=8)
    batch = _batch()
    torch.manual_seed(0)
    l2 = float(compute_field_flow_losses(m, batch, velocity_norm="l2")["loss_velocity"].detach())
    l1 = float(compute_field_flow_losses(m, batch, velocity_norm="l1")["loss_velocity"].detach())
    assert abs(l2 - l1) > 1e-8


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter
    from spectramr.infrastructure.training.strategies.field_flow_strategy import (
        FieldFlowStrategy,
    )
    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldFlowStrategy)
    strat.env = types.SimpleNamespace(generator=FieldVelocityUNet(width=8))

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the velocity model (train + val) ---


def test_contrast_id_threaded_train_and_validation() -> None:
    from spectramr.infrastructure.training.strategies.field_flow_strategy import (
        compute_field_flow_losses,
        integrate_field_flow,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_field_flow_losses(spy, batch)
    assert seen["cid"] is cid  # training forward threads it
    seen.clear()
    integrate_field_flow(
        spy, torch.rand(2, 1, 8, 8), 0.1, 7.0, n_steps=1, solver="euler", contrast_id=cid
    )
    assert seen["cid"] is cid  # validation integrator threads it
