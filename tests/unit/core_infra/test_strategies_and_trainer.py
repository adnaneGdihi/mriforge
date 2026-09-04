"""Unit tests for the training strategy templates and unified StepExecutor orchestration."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.step_executor import (
    OptimizationStepConfig,
    StepExecutor,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class TestStrategiesAndStepExecutor:
    """Verify strategy outputs and trainer orchestration logic like grad accumulation."""

    @pytest.fixture
    def mock_amp_policy(self):
        policy = MagicMock()
        policy.backward_and_step = MagicMock()
        return policy

    @pytest.fixture
    def mock_amp_helper(self):
        helper = MagicMock()
        helper.device_type = "cpu"
        helper.enabled = False
        helper.scaler = MagicMock()  # Mock the GradScaler
        return helper

    @pytest.fixture
    def dummy_optimizer(self):
        model = DummyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        # Mock zero_grad to count calls
        opt.zero_grad = MagicMock(wraps=opt.zero_grad)
        return opt

    def test_trainer_executes_step_configs_correctly(
        self, mock_amp_helper, mock_amp_policy, dummy_optimizer
    ):
        """Test that the StepExecutor properly unpacks and processes a list of step configs."""
        trainer = StepExecutor(
            amp_helper=mock_amp_helper,
            amp_policy=mock_amp_policy,
            gradient_accumulation_steps=1,
        )

        model = DummyModel()
        loss_tensor = torch.tensor(42.0, requires_grad=True)

        config: OptimizationStepConfig = {
            "name": "generator",
            "model": model,
            "optimizer": dummy_optimizer,
            "closure": lambda: loss_tensor,
        }

        # Execute
        losses = trainer.execute_step([config], epoch=1, global_step=0)

        # Asserts
        assert "generator_loss" in losses
        assert losses["generator_loss"].item() == 42.0

        # Verify optimizer zero_grad
        dummy_optimizer.zero_grad.assert_called_once_with(set_to_none=True)

        # Verify AMP Policy backward_and_step was called
        mock_amp_policy.backward_and_step.assert_called_once()

        # We explicitly enforce that the scaler is passed correctly
        call_kwargs = mock_amp_policy.backward_and_step.call_args.kwargs
        assert call_kwargs["loss"] is loss_tensor
        assert call_kwargs["optimizer"] is dummy_optimizer
        assert call_kwargs["scaler"] is mock_amp_helper.scaler
        assert call_kwargs["model"] is model
        assert call_kwargs["perform_step"] is True

    def test_trainer_gradient_accumulation_logic(
        self, mock_amp_helper, mock_amp_policy, dummy_optimizer
    ):
        """Test zero_grad and perform_step are clamped to gradient_accumulation boundaries."""
        trainer = StepExecutor(
            amp_helper=mock_amp_helper,
            amp_policy=mock_amp_policy,
            gradient_accumulation_steps=2,
        )

        loss_tensor = torch.tensor(10.0, requires_grad=True)

        config: OptimizationStepConfig = {
            "name": "accum_test",
            "model": DummyModel(),
            "optimizer": dummy_optimizer,
            "closure": lambda: loss_tensor,
        }

        # Pass 1: global_step 0 Modulo 2 != 1, so perform_step=False
        trainer.execute_step([config], epoch=1, global_step=0)

        # zero_grad is called on step 0
        assert dummy_optimizer.zero_grad.call_count == 1

        call_kwargs = mock_amp_policy.backward_and_step.call_args.kwargs
        assert call_kwargs["perform_step"] is False

        # Due to gradient accumulation = 2, the loss tensor should be scaled down by 1/2
        # loss / accum_steps = 10 / 2 = 5.0
        assert call_kwargs["loss"].item() == 5.0

        # Pass 2: global_step 1 Modulo 2 == 1, so perform_step=True
        trainer.execute_step([config], epoch=1, global_step=1)

        # zero_grad is NOT called on step 1 (accumulation is ongoing)
        assert dummy_optimizer.zero_grad.call_count == 1

        call_kwargs2 = mock_amp_policy.backward_and_step.call_args.kwargs
        assert call_kwargs2["perform_step"] is True

    def test_trainer_multiple_configs_sequential(
        self, mock_amp_helper, mock_amp_policy, dummy_optimizer
    ):
        """Mock the execution of sequential configurations (e.g., GAN Discriminator then Generator)."""
        trainer = StepExecutor(
            amp_helper=mock_amp_helper,
            amp_policy=mock_amp_policy,
            gradient_accumulation_steps=1,
        )

        d_loss = torch.tensor(1.0)
        g_loss = torch.tensor(2.0)

        configs = [
            {
                "name": "discriminator",
                "model": DummyModel(),
                "optimizer": dummy_optimizer,
                "closure": lambda: d_loss,
            },
            {
                "name": "generator",
                "model": DummyModel(),
                "optimizer": dummy_optimizer,
                "closure": lambda: g_loss,
            },
        ]

        losses = trainer.execute_step(configs, epoch=1, global_step=5)

        assert len(losses) == 2
        assert losses["discriminator_loss"].item() == 1.0
        assert losses["generator_loss"].item() == 2.0

        # Expected two separate backward/step executions
        assert mock_amp_policy.backward_and_step.call_count == 2
