"""Unit tests for TRELLISTrainingStrategy."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from mriforge.infrastructure.training.strategies.volumetric import TRELLISTrainingStrategy


@pytest.fixture
def mock_env():
    """Create a mock training environment (NEW API - env only)."""
    env = MagicMock()
    env.device = torch.device("cpu")

    # Mock generator returning 3D volume
    gen = MagicMock()
    gen.return_value = torch.randn(2, 1, 32, 64, 64)  # [B, C, D, H, W]
    gen.training = True
    gen.train = MagicMock()
    env.generator = gen
    env.models = {"generator": gen}

    # Mock optimizer
    opt_g = MagicMock()
    env.opt_g = opt_g
    env.optimizers = {"opt_g": opt_g}

    # Mock config
    env.config = MagicMock()
    env.config.optimization.optimizer.learning_rate = 1e-4
    # schema defaults; bare MagicMock fails StandardOptimizerStepper's raise-on-unknown
    env.config.optimization.gradient.clip.enabled = False
    # mixed_precision.py validates amp_dtype; a bare MagicMock fails it
    env.config.optimization.precision.dtype = "float32"
    env.config.optimization.gradient.clip.method = "norm"
    env.config.optimization.gradient.clip.value = 1.0
    env.config.model.model_type = "trellis"
    env.model_type = "trellis"

    return env


@pytest.fixture
def strategy(mock_env):
    """Instantiate TRELLISTrainingStrategy (NEW API - env only)."""
    # Patch UnifiedReconstructionLossComputer
    with patch(
        "mriforge.infrastructure.training.strategies.volumetric.UnifiedReconstructionLossComputer"
    ) as MockCalc:
        strategy = TRELLISTrainingStrategy(env=mock_env)
        return strategy


def test_initialization(strategy):
    """Test initialization."""
    assert strategy is not None
    assert strategy.loss_computer is not None


def test_compute_losses_impl_volumetric(strategy, mock_env):
    """Test volumetric loss computation."""
    input_batch = torch.randn(2, 1, 64, 64)  # 2D input
    target_batch = torch.randn(2, 1, 32, 64, 64)  # 3D target

    losses = strategy._compute_losses_impl(input_batch, target_batch, epoch=0)

    assert "g_total_loss" in losses
    # Defaults to L1 if no env losses
    assert "l1" in losses or "g_total_loss" in losses

    # Verify generator call
    mock_env.generator.assert_called_with(input_batch)


def test_compute_losses_impl_missing_optimizer(strategy, mock_env):
    """Test error when optimizer is missing."""
    mock_env.opt_g = None
    with pytest.raises(RuntimeError, match="Generator optimizer is required"):
        strategy._compute_losses_impl(torch.randn(1), torch.randn(1), 0)


def test_validation_step_metrics(strategy, mock_env):
    """Test validation metrics calculation."""
    input_batch = torch.randn(2, 1, 64, 64)
    target_batch = torch.randn(2, 1, 32, 64, 64)

    # Mock metrics adapter (optional)
    strategy.metrics_adapter = MagicMock()
    strategy.metrics_adapter.summarize.return_value = {"iou": 0.8}

    # validation_step now takes (input_batch, target_batch) — the leading
    # `batch` positional was removed when the validation API was unified.
    metrics = strategy.validation_step(input_batch, target_batch)

    assert "iou" in metrics
    assert "volume_mse" in metrics
    assert "volume_l1" in metrics
