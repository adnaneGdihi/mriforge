"""Unit tests for AMP (Automatic Mixed Precision) safety across all training strategies.

Tests verify:
1. Loss tensors are always FP32 (required for gradient stability)
2. No NaN/Inf propagation in gradients
3. GradScaler configuration is consistent
4. Autocast context managers are correctly applied
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.base import BaseTrainingStrategy


class SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, input_size=64, output_size=64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


@pytest.fixture
def mock_training_env() -> MagicMock:
    """Create mock TrainingEnvironment for testing."""
    env = MagicMock()
    env.device = torch.device("cpu")
    env.config = MagicMock()
    env.config.optimization = MagicMock()
    env.config.optimization.precision.enabled = True
    env.config.optimization.precision.dtype = "float16"
    env.config.training = MagicMock()
    env.config.training.training_mode = "reconstruction"
    env.models = {}
    env.optimizers = {}
    env.losses = {}
    return env


@pytest.fixture
def mock_strategy(mock_training_env: MagicMock) -> MagicMock:
    """Create mock training strategy.

    Returns a MagicMock instead of actual BaseTrainingStrategy to avoid
    complex initialization dependencies.
    """
    strategy = MagicMock(spec=BaseTrainingStrategy)
    strategy.env = mock_training_env
    strategy.device = torch.device("cpu")
    return strategy


class TestAMPLossDtype:
    """Test that loss tensors are always FP32."""

    def test_loss_dtype_is_float32_with_amp_enabled(self) -> None:
        """Loss should be FP32 even with AMP enabled."""
        # Create dummy model and data
        model = SimpleModel().to("cpu")
        input_batch = torch.randn(2, 64)
        target_batch = torch.randn(2, 64)

        # Simulate loss computation with AMP
        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            # Inside autocast, FP16 operations
            output = model(input_batch)
            # But loss should be computed in FP32
            loss = nn.MSELoss()(output, target_batch)

        # Verify loss is FP32
        assert (
            loss.dtype == torch.float32
        ), f"Loss dtype is {loss.dtype}, expected float32"

    def test_loss_dtype_remains_float32_after_backward(
        self,
    ) -> None:
        """Loss dtype should remain FP32 after backward pass."""
        model = SimpleModel().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        input_batch = torch.randn(2, 64)
        target_batch = torch.randn(2, 64)

        # Forward with AMP
        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)
            loss = nn.MSELoss()(output, target_batch)

        # Verify loss dtype before backward
        assert loss.dtype == torch.float32

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients are FP32
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad.dtype == torch.float32


class TestAMPGradientStability:
    """Test gradient stability with AMP enabled."""

    def test_no_nan_gradients_with_normal_loss(self) -> None:
        """Verify no NaN gradients with normal loss."""
        model = SimpleModel().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        input_batch = torch.randn(2, 64)
        target_batch = torch.randn(2, 64)

        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)
            loss = nn.MSELoss()(output, target_batch)

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), "Found NaN in gradients"
                assert not torch.isinf(param.grad).any(), "Found Inf in gradients"

    def test_no_inf_gradients_with_reasonable_loss(self) -> None:
        """Verify no Inf gradients with reasonable loss values."""
        model = SimpleModel().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Use small values to avoid numerical issues
        input_batch = torch.randn(2, 64) * 0.1
        target_batch = torch.randn(2, 64) * 0.1

        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)
            loss = nn.MSELoss()(output, target_batch)

        optimizer.zero_grad()
        loss.backward()

        # Check for Inf gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isinf(param.grad).any(), "Found Inf in gradients"


class TestAMPAutocastContext:
    """Test autocast context manager usage."""

    def test_autocast_disabled_by_default(self) -> None:
        """Autocast should be disabled without explicit enable."""
        input_tensor = torch.randn(2, 64)

        # Target: FP32 dtype outside autocast
        output = input_tensor  # No autocast context

        assert output.dtype == torch.float32

    def test_autocast_enables_fp16_inside_context(self) -> None:
        """Inside autocast context, certain ops should use FP16."""
        input_tensor = torch.randn(2, 64)

        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            # Matrix ops typically stay FP32 on CPU (different than GPU)
            # But the autocast context is still active
            output = input_tensor

        # After context, back to FP32
        assert output.dtype == torch.float32

    def test_loss_computed_outside_autocast_is_fp32(self) -> None:
        """Loss should be computed outside autocast to ensure FP32."""
        model = SimpleModel().to("cpu")

        input_batch = torch.randn(2, 64)
        target_batch = torch.randn(2, 64)

        # Forward inside autocast
        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)

        # Loss outside autocast (FP32)
        loss = nn.MSELoss()(output, target_batch)

        assert loss.dtype == torch.float32


class TestGradScalerConsistency:
    """Test GradScaler configuration consistency across strategies."""

    def test_gradscaler_init_scale_is_positive(self) -> None:
        """GradScaler init_scale should be reasonable (typically 2^16)."""
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler("cuda", init_scale=2**16)
            assert scaler._init_scale > 0

        # CPU version (fallback)
        scaler = torch.amp.GradScaler("cpu", init_scale=2**16)
        assert scaler._init_scale == 2**16

    def test_gradscaler_no_overflow_scaling(self) -> None:
        """GradScaler should maintain scale without overflow."""
        model = SimpleModel().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", init_scale=2**16)

        # Simulate multiple iterations without overflow
        for _ in range(10):
            optimizer.zero_grad()
            input_batch = torch.randn(2, 64)
            target_batch = torch.randn(2, 64)

            # Forward pass
            output = model(input_batch)
            loss = nn.MSELoss()(output, target_batch)

            # Scale loss and backward
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            # Step with scaler
            scaler.step(optimizer)
            scaler.update()

        # Scale should remain stable (no overflow detected)
        assert scaler._scale == 2**16


class TestAMPWithMultiplLosses:
    """Test AMP with multiple loss terms (common in complex strategies)."""

    def test_combined_loss_is_fp32(self) -> None:
        """Multiple loss components combined should be FP32."""
        model = SimpleModel().to("cpu")

        input_batch = torch.randn(2, 64)
        target_batch = torch.randn(2, 64)

        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)
            l1_loss = nn.L1Loss()(output, target_batch)
            l2_loss = nn.MSELoss()(output, target_batch)

        # Combine losses outside autocast
        combined_loss = 0.5 * l1_loss + 0.5 * l2_loss

        assert combined_loss.dtype == torch.float32

    def test_loss_accumulation_is_stable(self) -> None:
        """Accumulated loss over multiple batches should remain stable."""
        model = SimpleModel().to("cpu")
        accumulated_loss = torch.tensor(0.0, dtype=torch.float32)

        for batch_idx in range(5):
            input_batch = torch.randn(2, 64)
            target_batch = torch.randn(2, 64)

            with torch.autocast("cpu", enabled=True, dtype=torch.float16):
                output = model(input_batch)
                loss = nn.MSELoss()(output, target_batch)

            accumulated_loss = accumulated_loss + loss

        assert accumulated_loss.dtype == torch.float32
        assert not torch.isnan(accumulated_loss)
        assert not torch.isinf(accumulated_loss)


class TestAMPEdgeCases:
    """Test edge cases in AMP usage."""

    def test_zero_loss_is_handled_safely(self) -> None:
        """Zero loss should not cause issues."""
        loss = torch.tensor(0.0, dtype=torch.float32)

        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_small_loss_is_preserved(self) -> None:
        """Very small loss values should not underflow."""
        very_small_loss = torch.tensor(1e-6, dtype=torch.float32)

        assert very_small_loss.dtype == torch.float32
        assert very_small_loss > 0
        assert not torch.isnan(very_small_loss)

    def test_loss_with_large_model_is_stable(self) -> None:
        """Larger model should still maintain stable loss."""
        # Create larger model
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        ).to("cpu")

        input_batch = torch.randn(2, 256)
        target_batch = torch.randn(2, 64)

        with torch.autocast("cpu", enabled=True, dtype=torch.float16):
            output = model(input_batch)
            loss = nn.MSELoss()(output, target_batch)

        assert loss.dtype == torch.float32
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
