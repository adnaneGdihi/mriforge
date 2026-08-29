"""Unit tests ensuring loss functions are AMP and backward-pass safe."""

import pytest
import torch

from mriforge.models.losses.complex_losses import ComplexL1Loss, WeightedKSpaceL1Loss


class TestComplexLossesAMPSafety:
    """Ensure complex loss modules return real-valued scalar tensors."""

    @pytest.fixture
    def complex_predictions(self):
        """Mock complex predictions with requires_grad=True for AMP checks."""
        # [B, C, H, W]
        tensor = torch.randn(2, 1, 32, 32, dtype=torch.complex64)
        tensor.requires_grad_(True)
        return tensor

    @pytest.fixture
    def complex_targets(self):
        """Mock complex targets without gradients."""
        return torch.randn(2, 1, 32, 32, dtype=torch.complex64)

    @pytest.fixture
    def real_channel_predictions(self):
        """Mock stacked real predictions [B, C*2, H, W]."""
        tensor = torch.randn(2, 2, 32, 32, dtype=torch.float32)
        tensor.requires_grad_(True)
        return tensor

    @pytest.fixture
    def real_channel_targets(self):
        """Mock stacked real targets [B, C*2, H, W]."""
        return torch.randn(2, 2, 32, 32, dtype=torch.float32)

    def test_complex_l1_native_complex_inputs(
        self, complex_predictions, complex_targets
    ):
        """Test ComplexL1Loss safely handles native complex inputs."""
        criterion = ComplexL1Loss(reduction="mean")
        loss = criterion(complex_predictions, complex_targets)

        assert not torch.is_complex(loss), "Loss must be a real tensor for AMP backward"
        assert loss.dim() == 0, "Loss must be a scalar"
        assert loss.requires_grad, "Loss must maintain gradient graph"

        # Test backward pass to ensure no 'grad can be implicitly created only for real...'
        loss.backward()
        assert complex_predictions.grad is not None

    def test_complex_l1_stacked_real_inputs(
        self, real_channel_predictions, real_channel_targets
    ):
        """Test ComplexL1Loss safely handles stacked [B, 2, H, W] inputs."""
        criterion = ComplexL1Loss(reduction="mean")
        loss = criterion(real_channel_predictions, real_channel_targets)

        assert not torch.is_complex(loss)
        assert loss.dim() == 0
        assert loss.requires_grad

        loss.backward()
        assert real_channel_predictions.grad is not None

    def test_weighted_kspace_l1_native_complex(
        self, complex_predictions, complex_targets
    ):
        """Test WeightedKSpaceL1Loss with complex inputs."""
        criterion = WeightedKSpaceL1Loss()
        loss = criterion(complex_predictions, complex_targets)

        assert not torch.is_complex(loss)
        assert loss.dim() == 0
        assert loss.requires_grad

        loss.backward()
        assert complex_predictions.grad is not None

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_amp_scaler_compatibility(self, dtype):
        """Ensure loss explicitly works inside an autocast region and can be scaled."""
        criterion = ComplexL1Loss(reduction="mean")
        preds = torch.randn(2, 2, 16, 16, dtype=torch.float32, requires_grad=True)
        targets = torch.randn(2, 2, 16, 16, dtype=torch.float32)

        scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

        # Simulate AMP Forward
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype
        ):
            loss = criterion(preds, targets)

        assert not torch.is_complex(loss)

        # Simulate AMP Backward (this crashes if loss is complex)
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        assert preds.grad is not None
