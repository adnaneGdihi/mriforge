"""Regression tests for the 2026-06 pipeline_strategy audit fixes.

Covers the three BROKEN_STRATEGY_ONLY findings against
``spectramr.infrastructure.training.strategies.pipeline_strategy``:

1. [high]   Per-stage loss failures were silently swallowed by a
   ``try/except Exception: logger.warning(...)`` in ``_compute_stage_losses``,
   so a broken configured loss trained on a subset of the declared objective
   (pitfall #9/#10). The fix removes the catch so the exception propagates.
2. [medium] A stage config-merge failure silently fell back to the global
   config in ``_build_stage_losses`` (``except Exception: merged_config =
   self.config``), discarding the stage's declared overrides without any
   indication. The fix raises ``RuntimeError`` instead.
3. [medium] ``_step_stage_optimizers`` evaluated ``float(stage_total.item())``
   in a per-step debug log (Python evaluates logger args eagerly), forcing a
   GPU sync every step (performance.md: no ``.item()`` in the loop). The fix
   drops the value from the log.

Each test builds a minimally-populated strategy via ``object.__new__`` (the
sibling pattern in ``test_pipeline_params_registered.py``) so only the fixed
method is exercised — no full ``__init__`` / config plumbing is required.
"""

from __future__ import annotations

import inspect
import types

from tests.utils.config_block_stub import block_stub

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)


class _RaisingLoss(nn.Module):
    """A loss module whose forward always raises (simulates a broken loss)."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("loss kernel exploded")


# ---------------------------------------------------------------------------
# Finding 1 — per-stage loss failure must propagate, not be swallowed
# ---------------------------------------------------------------------------


class TestStageLossFailurePropagates:
    def _strategy_with_loss(self, loss: nn.Module) -> MultiTrainingStrategy:
        s = object.__new__(MultiTrainingStrategy)
        s._stage_losses = {"stage_a": {"l1": loss}}
        s._stage_loss_weights = {"stage_a": {"l1": 1.0}}
        return s

    def test_broken_loss_raises(self) -> None:
        s = self._strategy_with_loss(_RaisingLoss())
        pred = torch.zeros(2, 1, 4, 4)
        tgt = torch.zeros(2, 1, 4, 4)
        with pytest.raises(RuntimeError, match="loss kernel exploded"):
            s._compute_stage_losses("stage_a", pred, tgt)

    def test_working_loss_still_aggregates(self) -> None:
        # The happy path must remain intact after removing the try/except.
        s = self._strategy_with_loss(nn.L1Loss())
        pred = torch.ones(2, 1, 4, 4)
        tgt = torch.zeros(2, 1, 4, 4)
        out = s._compute_stage_losses("stage_a", pred, tgt)
        assert "stage_a/l1" in out
        assert torch.allclose(out["stage_a/l1"], torch.tensor(1.0))

    def test_source_has_no_swallowing_except(self) -> None:
        # Guard against re-introduction of the silent fallback.
        src = inspect.getsource(MultiTrainingStrategy._compute_stage_losses)
        assert "loss '%s' failed" not in src
        assert "except Exception" not in src


# ---------------------------------------------------------------------------
# Finding 2 — config-merge failure must raise, not fall back to global config
# ---------------------------------------------------------------------------


class TestConfigMergeFailureRaises:
    def test_merge_failure_raises_runtimeerror(self) -> None:
        s = object.__new__(MultiTrainingStrategy)

        class _BoomStageCfg:
            name = "stage_a"
            training_state = types.SimpleNamespace(
                freeze_all=False, inject_lora=False, unfreeze_modules=[]
            )
            stage_config = object()

        s.stage_configs = {"stage_a": _BoomStageCfg()}
        s._stage_losses = {}
        s._stage_loss_weights = {}
        s._stage_early_stopping = {}
        s.device = torch.device("cpu")
        # A sentinel that must NEVER be used as the fallback merged_config.
        s.config = object()

        def _boom(_stage_cfg: object) -> None:
            raise ValueError("merge validation failed")

        s._merge_stage_config = _boom  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Stage 'stage_a' config merge failed"):
            s._build_stage_losses()

    def test_source_has_no_global_config_fallback(self) -> None:
        src = inspect.getsource(MultiTrainingStrategy._build_stage_losses)
        assert "merged_config = self.config" not in src
        assert "using global" not in src


# ---------------------------------------------------------------------------
# Finding 3 — no .item() GPU sync in the per-step stage-optimizer hot path
# ---------------------------------------------------------------------------


class TestNoItemSyncInStageStep:
    def test_step_does_not_sync_with_item(self, monkeypatch) -> None:
        # Force DEBUG so the previously-offending debug log actually formats its
        # args (Python evaluates them eagerly regardless of level). Any surviving
        # ``stage_total.item()`` would now be invoked — trip a counter if so.
        import logging as _logging

        from spectramr.infrastructure.training.strategies import (
            pipeline_strategy as _ps,
        )

        monkeypatch.setattr(_ps.logger, "level", _logging.DEBUG, raising=False)
        _ps.logger.setLevel(_logging.DEBUG)

        calls = {"item": 0}
        orig_item = torch.Tensor.item

        def _counting_item(self: torch.Tensor, *a: object, **k: object) -> float:
            calls["item"] += 1
            return orig_item(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, "item", _counting_item, raising=True)

        s = object.__new__(MultiTrainingStrategy)
        s.device = torch.device("cpu")

        stage = nn.Linear(4, 4)
        opt = torch.optim.SGD(stage.parameters(), lr=0.01)
        s._stage_optimizers = {"stage_a": opt}

        target = torch.zeros(2, 4)
        pred = stage(torch.randn(2, 4))
        assert pred.shape == target.shape
        state = {"stage_a": pred}

        # Real, grad-bearing per-stage loss (backward/step must succeed).
        def _fake_compute(
            stage_name: str, prediction: torch.Tensor, tgt: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {f"{stage_name}/mse": ((prediction - tgt) ** 2).mean()}

        s._compute_stage_losses = _fake_compute  # type: ignore[method-assign]

        stepped = s._step_stage_optimizers(state, target, epoch=0)
        assert "stage_a/mse" in stepped
        # The detached logging-only loss does NOT use .item(); the per-step path
        # must not force a host transfer.
        assert calls["item"] == 0, "._step_stage_optimizers still calls .item()"

    def test_source_has_no_item_call_in_step(self) -> None:
        # Strip comment lines, then assert no real ``.item()`` call survives.
        src = inspect.getsource(MultiTrainingStrategy._step_stage_optimizers)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert ".item()" not in code
        assert "loss=%.6f" not in code


# ---------------------------------------------------------------------------
# [SRV-001] — per-stage optimizer dispatch routes through the registry and
#             RAISES on an unknown optimizer_type (no silent AdamW fallback)
# ---------------------------------------------------------------------------


def _stage_cfg_with_optimizer(opt_type: str) -> object:
    """Minimal stage config exposing ``stage_config.optimization``.

    Mirrors the attribute path read by ``_build_stage_optimizers``:
    ``stage_cfg.stage_config.optimization.{learning_rate,optimizer_type,
    weight_decay}``.
    """
    # `learning_rate`/`optimizer_type`/`weight_decay` all live under
    # `optimization.optimizer.*` now; the flat namespace left the reader on
    # schema defaults.
    optimization = block_stub(
        "optimization",
        learning_rate=1e-3,
        optimizer_type=opt_type,
        weight_decay=0.0,
    )
    stage_config = types.SimpleNamespace(optimization=optimization)
    return types.SimpleNamespace(stage_config=stage_config)


class TestStageOptimizerRaisesOnUnknownType:
    def _strategy_for_opt_type(self, opt_type: str) -> MultiTrainingStrategy:
        s = object.__new__(MultiTrainingStrategy)
        s.stage_configs = {"stage_a": _stage_cfg_with_optimizer(opt_type)}
        s.multi_stages = {"stage_a": nn.Linear(4, 4)}
        s._stage_optimizers = {}
        return s

    def test_unknown_optimizer_type_raises(self) -> None:
        s = self._strategy_for_opt_type("__bogus__")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            s._build_stage_optimizers()

    def test_known_optimizer_type_builds(self) -> None:
        # The happy path must remain intact after routing through the registry.
        s = self._strategy_for_opt_type("adam")
        s._build_stage_optimizers()
        opt = s._stage_optimizers["stage_a"]
        assert isinstance(opt, torch.optim.Adam)

    def test_source_routes_through_create_optimizer(self) -> None:
        # Guard against re-introduction of the silent if/elif/else AdamW fallback.
        src = inspect.getsource(MultiTrainingStrategy._build_stage_optimizers)
        assert "create_optimizer(" in src
        assert "torch.optim.AdamW(" not in src


# ---------------------------------------------------------------------------
# [SRV-002] — per-step stage zero_grad must pass set_to_none=True (NN#9)
# ---------------------------------------------------------------------------


class TestStageOptimizerZeroGradSetToNone:
    def test_zero_grad_called_with_set_to_none(self, monkeypatch) -> None:
        s = object.__new__(MultiTrainingStrategy)
        s.device = torch.device("cpu")

        stage = nn.Linear(4, 4)
        opt = torch.optim.SGD(stage.parameters(), lr=0.01)
        s._stage_optimizers = {"stage_a": opt}

        captured: dict[str, object] = {}
        orig_zero_grad = opt.zero_grad

        def _spy_zero_grad(*a: object, set_to_none: bool = True, **k: object) -> None:
            captured["set_to_none"] = set_to_none
            return orig_zero_grad(set_to_none=set_to_none)

        monkeypatch.setattr(opt, "zero_grad", _spy_zero_grad, raising=True)

        target = torch.zeros(2, 4)
        pred = stage(torch.randn(2, 4))
        state = {"stage_a": pred}

        def _fake_compute(
            stage_name: str, prediction: torch.Tensor, tgt: torch.Tensor
        ) -> dict[str, torch.Tensor]:
            return {f"{stage_name}/mse": ((prediction - tgt) ** 2).mean()}

        s._compute_stage_losses = _fake_compute  # type: ignore[method-assign]

        s._step_stage_optimizers(state, target, epoch=0)
        assert captured.get("set_to_none") is True

    def test_source_passes_set_to_none(self) -> None:
        # Guard against regression to bare ``optimizer.zero_grad()``.
        src = inspect.getsource(MultiTrainingStrategy._step_stage_optimizers)
        assert "optimizer.zero_grad(set_to_none=True)" in src
        # No bare zero_grad() with empty args survives.
        assert "optimizer.zero_grad()" not in src
