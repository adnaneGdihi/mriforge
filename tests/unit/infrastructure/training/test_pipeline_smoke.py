"""Comprehensive smoke tests for the multi-stage pipeline configuration and strategy.

Validates the full lifecycle:
1. Schema parsing from complex Python dicts
2. Strategy instantiation and DI container integration
3. Dynamic component building (losses, optimizers, metrics, early stopping)
4. Forward pass, recursive action interpretation, optimizer step simulation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from mriforge.config.schemas.training.base import TrainingStrategyConfigSchema
from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.training.strategies.pipeline_strategy import (
    MultiTrainingStrategy,
)

# ---------------------------------------------------------------------------
# Dummy Classes for Strategy
# ---------------------------------------------------------------------------


class _DummyMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, y_pred, y) -> float:
        return 0.42


# ---------------------------------------------------------------------------
# Multi-stage Config Generation
# ---------------------------------------------------------------------------


def _get_dummy_multi_config() -> dict:
    """Returns a full multi-stage configuration dictionary for smoke testing."""
    return {
        "strategy_class": "multi",
        "multi": {
            "stages": [
                {
                    "name": "stage_a",
                    "model_type": "standard_unet",
                    "training_state": {
                        "freeze_all": False,
                    },
                    "stage_config": {
                        "loss": {
                            "reconstruction": {"enable_l1": True, "lambda_l1": 1.0}
                        },
                        "optimization": {
                            "optimizer_type": "adamw",
                            "learning_rate": 0.001,
                        },
                    },
                },
                {
                    "name": "stage_b",
                    "model_type": "standard_unet",
                    "training_state": {
                        "freeze_all": False,
                    },
                    "stage_config": {
                        "loss": {
                            "reconstruction": {"enable_l2": True, "lambda_l2": 2.0}
                        },
                        "early_stopping": {
                            "enabled": True,
                            "patience": 2,
                            "metric": "val_stage_b/l2",
                        },
                    },
                },
            ],
            "routines": {
                "train": [
                    {
                        "action": "call",
                        "node": "stage_a",
                        "inputs": {"x": "batch.input"},
                        # The output state key must equal the stage name:
                        # _aggregate_stage_losses / _step_stage_optimizers look
                        # up per-stage outputs from `state` by stage name.
                        "outputs": {"stage_a": "output"},
                    },
                    {
                        "action": "call",
                        "node": "stage_b",
                        "inputs": {"x": "state.stage_a"},
                        "outputs": {"stage_b": "output"},
                    },
                ]
            },
            "evaluation": {
                "metrics": {
                    "dummy_score": {
                        "target": "tests.unit.infrastructure.training.test_pipeline_smoke._DummyMetric",
                        "params": {"custom_param": 10},
                        "inputs": {"y_pred": "state.stage_b", "y": "batch.target"},
                    }
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


class TestMultiStageSmoke:
    def test_schema_instantiation(self) -> None:
        """Smoke test 1: Validate complex nested dict generates valid Pydantic schema."""
        cfg_dict = _get_dummy_multi_config()
        schema = TrainingStrategyConfigSchema.model_validate(cfg_dict)

        assert schema.strategy_class == "multi"
        assert schema.multi is not None
        assert len(schema.multi.stages) == 2

        # Verify nested stage environments
        stage_a = schema.multi.stages[0]
        assert stage_a.stage_config.loss.reconstruction.enable_l1
        assert stage_a.stage_config.optimization.optimizer.learning_rate == 0.001

        stage_b = schema.multi.stages[1]
        assert stage_b.stage_config.early_stopping.patience == 2

    def test_strategy_lifecycle(self) -> None:
        """Smoke test 2: Full strategy build, forward pass, aggregation, and validation.

        Rewritten against the current ``MultiTrainingStrategy``: stage models
        are built via the real ``GeneratorBuilder`` + model registry (no
        ``ModelFactory`` — that seam no longer exists), so this uses a real
        registered tiny ``standard_unet`` per stage instead of mocking model
        construction. Early stopping is verified against the actual
        ``_es_best``/``_es_wait`` + per-parameter ``requires_grad`` freeze
        mechanism (there is no ``_early_stopping_trackers`` object).
        """
        # 1. Build a full, valid TrainingSettings wrapping the multi-stage config.
        settings = TrainingSettings.model_validate(
            {
                "device": "cpu",
                "model": {
                    "model_type": "standard_unet",
                    "in_channels": 1,
                    "out_channels": 1,
                },
                "data": {
                    "dataset_type": "synthetic",
                    "patch_size": [64, 64],
                    "batch_size": 2,
                },
                "training": _get_dummy_multi_config(),
                "optimization": {"learning_rate": 1e-4},
                "logging": {},
            }
        )

        # 2. Minimal real TrainingEnvironment stand-in (no ModelFactory mock —
        # stage models are built for real by MultiTrainingStrategy itself).
        # Was ``mocker.MagicMock()``. ``pytest-mock`` is not declared in
        # pyproject.toml at ALL and is absent from the project venv, so the
        # ``mocker`` fixture never resolved on any machine that installed the
        # project's own extras -- this was an undeclared dependency, not the
        # cluster gap it looked like on job 8004252. stdlib is the fix; adding
        # the dep would make one test file dictate a new runtime requirement.
        env = MagicMock()
        env.config = settings
        env.device = torch.device("cpu")
        env.opt_g = None  # stage_b has no dedicated optimizer; not exercised here.

        # 3. Instantiate Strategy
        strategy = MultiTrainingStrategy(env=env, device=torch.device("cpu"))

        # Check introspection outputs
        assert len(strategy.multi_stages) == 2
        assert "stage_a" in strategy._stage_losses
        assert "stage_b" in strategy._stage_losses
        assert "stage_a" in strategy._stage_optimizers
        assert "stage_b" not in strategy._stage_optimizers  # Was not configured
        assert "stage_b" in strategy._stage_early_stopping
        assert "dummy_score" in strategy._eval_metrics

        # 4. Forward Pass / Compute Losses — image-shaped batch for standard_unet.
        bsz = 2
        input_t = torch.randn(bsz, 1, 64, 64, requires_grad=True)
        target_t = torch.randn(bsz, 1, 64, 64)

        # We must intercept zero_grad to prove stage_a optimizer was used
        stage_a_opt = strategy._stage_optimizers["stage_a"]
        # ``patch.object(..., wraps=...)`` is the stdlib equivalent of
        # ``mocker.spy``: it records the calls AND calls through to the real
        # method, so the optimizer still steps. A plain Mock would record the
        # calls while silently disabling the step this test exists to prove.
        with (
            patch.object(
                stage_a_opt, "zero_grad", wraps=stage_a_opt.zero_grad
            ) as spy_zero_grad,
            patch.object(stage_a_opt, "step", wraps=stage_a_opt.step) as spy_step,
        ):
            losses = strategy._compute_losses_impl(input_t, target_t, epoch=0)

        # Verification
        assert "stage_a/l1" in losses
        assert "stage_b/l2" in losses
        assert "g_total_loss" in losses

        # Fix #1 Verify: stage_a was stepped internally
        spy_zero_grad.assert_called_once()
        spy_step.assert_called_once()

        # 5. Validation Step
        val_metrics = strategy.validation_step(input_t, target_t)

        # Fix #2 Verify: Eval metric dummy_score is integrated
        assert "val_dummy_score" in val_metrics
        assert val_metrics["val_dummy_score"] == 0.42
        assert "val_stage_b/l2" in val_metrics

        # 6. Early Stopping Step — real mechanism: non-improving metric past
        # `patience` freezes the stage's parameters (no tracker object).
        stage_b_model = strategy.multi_stages["stage_b"]
        assert any(p.requires_grad for p in stage_b_model.parameters())

        # Simulate bad validation performance over 3 epochs (patience=2)
        metrics = {"val_stage_b/l2": 10.0}
        strategy.on_epoch_end(epoch=1, metrics=metrics)
        strategy.on_epoch_end(epoch=2, metrics=metrics)
        strategy.on_epoch_end(epoch=3, metrics=metrics)

        assert all(not p.requires_grad for p in stage_b_model.parameters())
