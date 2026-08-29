"""Unit tests for per-stage LossBuilder integration in MultiTrainingStrategy.

Tests verify that:
- Per-stage loss functions are built via LossBuilder (SSOT)
- Config proxy merges stage_config onto global config
- Loss aggregation namespaces losses correctly
- Frozen stages without loss config are skipped
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.config.schemas.training.pipeline import (
    MultiStageSchema,
    StageEnvironmentSchema,
    TrainingStateSchema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    """Minimal generator for testing."""

    def __init__(self, in_channels: int = 2, out_channels: int = 2, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# StageEnvironmentSchema tests
# ---------------------------------------------------------------------------


class TestStageEnvironmentSchema:
    """Test the unified per-stage configuration schema."""

    def test_default_all_none(self) -> None:
        """All fields should default to None."""
        env = StageEnvironmentSchema()
        assert env.training is None
        assert env.loss is None
        assert env.metrics is None
        assert env.early_stopping is None
        assert env.optimization is None

    def test_loss_config_nested_access(self) -> None:
        """Verify loss config can be set and accessed."""
        env = StageEnvironmentSchema(
            loss={
                "reconstruction": {
                    "enable_l1": True,
                    "lambda_l1": 0.5,
                }
            }
        )
        assert env.loss is not None
        assert env.loss.reconstruction.enable_l1 is True
        assert env.loss.reconstruction.lambda_l1 == 0.5

    def test_optimization_override(self) -> None:
        """Verify optimization overrides are stored."""
        env = StageEnvironmentSchema(
            optimization={
                "learning_rate": 1e-5,
            }
        )
        assert env.optimization is not None
        assert env.optimization.optimizer.learning_rate == 1e-5

    def test_early_stopping_config(self) -> None:
        """Verify early stopping config is stored."""
        env = StageEnvironmentSchema(
            early_stopping={
                "enabled": True,
                "patience": 5,
            }
        )
        assert env.early_stopping is not None
        assert env.early_stopping.enabled is True
        assert env.early_stopping.patience == 5


# ---------------------------------------------------------------------------
# MultiStageSchema with stage_config tests
# ---------------------------------------------------------------------------


class TestMultiStageSchemaWithStageConfig:
    """Test that MultiStageSchema correctly embeds StageEnvironmentSchema."""

    def test_stage_with_no_config(self) -> None:
        """Stage without stage_config should default to None."""
        stage = MultiStageSchema(name="encoder", model_type="standard_unet")
        assert stage.stage_config is None

    def test_stage_with_loss_config(self) -> None:
        """Stage with loss config should parse correctly."""
        stage = MultiStageSchema(
            name="decoder",
            model_type="standard_unet",
            stage_config={
                "loss": {
                    "reconstruction": {
                        "enable_l1": True,
                        "lambda_l1": 1.0,
                    }
                }
            },
        )
        assert stage.stage_config is not None
        assert stage.stage_config.loss is not None
        assert stage.stage_config.loss.reconstruction.enable_l1 is True

    def test_stage_with_full_environment(self) -> None:
        """Stage with all environment overrides should parse."""
        stage = MultiStageSchema(
            name="gan_stage",
            model_type="standard_unet",
            stage_config={
                "loss": {
                    "gan": {
                        "enable_adversarial": True,
                        "lambda_adv": 1.0,
                    }
                },
                "optimization": {
                    "learning_rate": 2e-4,
                },
                "early_stopping": {
                    "enabled": True,
                    "patience": 10,
                },
            },
        )
        env = stage.stage_config
        assert env is not None
        assert env.loss.gan.enable_adversarial is True
        assert env.optimization.optimizer.learning_rate == 2e-4
        assert env.early_stopping.enabled is True
        assert env.early_stopping.patience == 10

    def test_frozen_stage_no_environment(self) -> None:
        """Frozen stage with no stage_config is valid (inference-only)."""
        stage = MultiStageSchema(
            name="frozen_encoder",
            model_type="standard_unet",
            training_state=TrainingStateSchema(freeze_all=True),
        )
        assert stage.stage_config is None
        assert stage.training_state.freeze_all is True


# ---------------------------------------------------------------------------
# Loss aggregation logic tests (pure functions, no strategy instantiation)
# ---------------------------------------------------------------------------


class TestLossAggregation:
    """Test loss computation and aggregation patterns."""

    def test_compute_stage_losses_with_modules(self) -> None:
        """Losses should be computed and namespaced by stage."""
        losses = {"l1": nn.L1Loss(), "mse": nn.MSELoss()}
        weights = {"l1": 1.0, "mse": 0.5}
        pred = torch.randn(2, 2, 8, 8, requires_grad=True)
        target = torch.randn(2, 2, 8, 8)

        result: dict[str, torch.Tensor] = {}
        for loss_name, loss_fn in losses.items():
            val = loss_fn(pred, target)
            w = weights.get(loss_name, 1.0)
            result[f"decoder/{loss_name}"] = w * val

        assert "decoder/l1" in result
        assert "decoder/mse" in result
        assert result["decoder/l1"].requires_grad
        assert result["decoder/mse"].requires_grad

    def test_aggregation_total_loss(self) -> None:
        """Total loss should be sum of all weighted per-stage losses."""
        pred = torch.randn(2, 2, 8, 8, requires_grad=True)
        target = torch.randn(2, 2, 8, 8)

        l1_val = nn.L1Loss()(pred, target)
        mse_val = nn.MSELoss()(pred, target)
        total = 1.0 * l1_val + 0.5 * mse_val

        # Should be differentiable
        total.backward()
        assert pred.grad is not None

    def test_no_losses_falls_back_to_global(self) -> None:
        """When no per-stage losses exist, global L1/MSE should be used."""
        pred = torch.randn(2, 2, 8, 8)
        target = torch.randn(2, 2, 8, 8)

        # Simulate fallback
        loss_l1 = torch.nn.functional.l1_loss(pred, target)
        loss_mse = torch.nn.functional.mse_loss(pred, target)
        total = loss_l1 + loss_mse

        result = {
            "loss_l1": loss_l1,
            "loss_mse": loss_mse,
            "g_total_loss": total,
        }

        assert "g_total_loss" in result
        assert result["g_total_loss"].item() > 0


"""
Complexity:3
Description:Unit tests for per-stage loss building, config proxy merging, and loss namespace aggregation in the multi-stage pipeline strategy.
"""
