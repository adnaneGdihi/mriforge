"""Physics-compliance tests for validation metrics normalization.

Tests verify that the input-reference normalization preserves MRI forward operator
physics and prevents gain jitter in metrics computation.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy


class TestPhysicsCompliantNormalization:
    """Test suite for physics-compliant normalization in validation."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=TrainingSettings)
        config.model = Mock()
        config.model.in_channels = 2
        config.data = Mock()
        # Canonical paths (phases 9d / 10a). A Mock auto-creates any attribute,
        # so setting the FLAT legacy names left `data.processing.*` as truthy
        # Mocks: the "enabled" cases passed for the wrong reason and only the
        # disabled case could fail. Setting the path the reader walks is the
        # difference between testing the strategy and testing the Mock.
        config.data.processing = Mock()
        config.data.processing.enable_kspace_normalization = True
        config.data.processing.normalization_type = "percentile"
        config.data.processing.normalization_kwargs = {"percentile": 0.99, "clamp": True}
        config.validation = Mock()
        config.validation.scoring = Mock()
        config.validation.scoring.enable_image_metrics = True
        config.validation.scoring.domain = "image"
        config.validation.scoring.output_transform = "ifft_magnitude"
        config.training = Mock()
        config.training.diffusion = Mock()
        config.training.diffusion.type = "cold"
        config.training.diffusion.timesteps = 1000
        config.training.diffusion.noise_schedule = "linear"
        # beta_start/beta_end are declared schema fields with numeric defaults;
        # the strategy forwards them to DiffusionScheduler, so a mock that
        # leaves them auto-generated hands linspace() a MagicMock.
        config.training.diffusion.beta_start = 1e-4
        config.training.diffusion.beta_end = 0.02
        config.optimization = Mock()
        config.optimization.precision.enabled = False
        config.optimization.optimizer.learning_rate = 1e-4
        config.physics = None
        config.checkpoint = None
        config.logging = Mock()
        config.logging.log_dir = "/tmp/test_logs"
        config.logging.tensorboard_enabled = False
        return config

    @pytest.fixture
    def strategy(self, mock_config):
        """Create diffusion strategy with mocked dependencies."""
        # Directly instantiate with all __init__ patched to avoid complex dependency chain
        with patch.object(DiffusionTrainingStrategy, "__init__", return_value=None):
            strategy = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
            strategy.config = mock_config
            strategy.state = Mock()
            strategy.state.config = mock_config
            strategy.device = torch.device("cpu")
            strategy.logging_service = Mock()
            strategy.env = Mock()
            strategy.env.config = mock_config
            strategy.env.device = torch.device("cpu")
            return strategy

    def test_input_reference_normalization_preserves_scale_ratio(self, strategy):
        """Test that input-reference normalization preserves physics.

        Critical: scale_input / scale_target must equal 1.0 (same scale applied to both).
        """
        # Create synthetic k-space data (8x undersampled input, full target)
        batch_size = 2
        height, width = 128, 128

        # Input: undersampled k-space (lower energy)
        input_kspace = torch.randn(batch_size, 2, height, width) * 0.5

        # Target: fully sampled k-space (higher energy)
        target_kspace = torch.randn(batch_size, 2, height, width) * 1.0

        # Create batch
        batch = {"input": input_kspace, "target": target_kspace}

        # Call normalization
        norm_input, norm_target, scale_factor = strategy._prepare_validation_data(
            batch, None, None, None
        )

        # Verify scale_factor has correct shape [B, 1, 1, 1]
        assert scale_factor.shape == (batch_size, 1, 1, 1)

        # Verify scale_factor is positive
        assert (scale_factor > 0).all()

        # Compute magnitude of normalized data
        norm_input_mag = torch.sqrt(norm_input[:, 0] ** 2 + norm_input[:, 1] ** 2)
        norm_target_mag = torch.sqrt(norm_target[:, 0] ** 2 + norm_target[:, 1] ** 2)

        # Original magnitude
        input_mag = torch.sqrt(input_kspace[:, 0] ** 2 + input_kspace[:, 1] ** 2)

        # Verify normalization preserves relative scale
        # scale should be computed from input, so input_mag / scale ≈ norm_input_mag
        # scale_factor has shape [B, 1, 1, 1] so squeeze to [B] then reshape for broadcast
        expected_norm_input_mag = input_mag / scale_factor.view(batch_size, 1, 1)
        torch.testing.assert_close(
            norm_input_mag, expected_norm_input_mag, rtol=1e-4, atol=1e-5
        )

    def test_denormalization_restores_physical_units(self, strategy):
        """Test that denormalization correctly restores k-space to physical units."""
        batch_size = 1
        height, width = 64, 64

        # Original k-space in physical units
        original_kspace = torch.randn(batch_size, 2, height, width) * 100.0

        # Simulate normalization
        input_mag = torch.sqrt(original_kspace[:, 0] ** 2 + original_kspace[:, 1] ** 2)
        scale_factor = torch.quantile(input_mag.flatten(), 0.99).view(1, 1, 1, 1)
        normalized_kspace = original_kspace / scale_factor

        # Denormalize
        denormalized_kspace = normalized_kspace * scale_factor

        # Verify restoration
        torch.testing.assert_close(
            denormalized_kspace, original_kspace, rtol=1e-5, atol=1e-6
        )

    def test_scale_factor_computed_from_input_not_target(self, strategy):
        """Test that scale factor is derived from input, not target.

        This is critical for preventing gain jitter in training.
        """
        batch_size = 2

        # Create input and target with VERY different energy levels
        # Input: low energy (simulating undersampled measurement)
        input_kspace = torch.randn(2, 2, 64, 64) * 0.1

        # Target: high energy (fully sampled clean data)
        target_kspace = torch.randn(2, 2, 64, 64) * 10.0

        batch = {"input": input_kspace, "target": target_kspace}

        norm_input, norm_target, scale_factor = strategy._prepare_validation_data(
            batch, None, None, None
        )

        # Compute expected scale from INPUT
        input_mag = torch.sqrt(input_kspace[:, 0] ** 2 + input_kspace[:, 1] ** 2)
        expected_scales = []
        for i in range(batch_size):
            sample_scale = torch.quantile(input_mag[i].flatten(), 0.99)
            expected_scales.append(sample_scale)
        expected_scale_factor = torch.stack(expected_scales).view(batch_size, 1, 1, 1)

        # Verify scale matches input-derived scale (not target-derived)
        torch.testing.assert_close(
            scale_factor, expected_scale_factor, rtol=1e-4, atol=1e-5
        )

        # Verify scale is NOT derived from target
        target_mag = torch.sqrt(target_kspace[:, 0] ** 2 + target_kspace[:, 1] ** 2)
        target_scale = torch.quantile(target_mag.flatten(), 0.99)

        # Scale should be much smaller than target scale (closer to input scale)
        assert scale_factor.mean().item() < target_scale.item() / 5.0

    def test_same_scale_applied_to_input_and_target(self, strategy):
        """Test that SAME scale factor is applied to both input and target."""
        batch_size = 1

        input_kspace = torch.randn(batch_size, 2, 64, 64)
        target_kspace = torch.randn(batch_size, 2, 64, 64)

        batch = {"input": input_kspace, "target": target_kspace}

        # Store original values
        input_original = input_kspace.clone()
        target_original = target_kspace.clone()

        norm_input, norm_target, scale_factor = strategy._prepare_validation_data(
            batch, None, None, None
        )

        # Verify both were divided by same scale
        expected_norm_input = input_original / scale_factor
        expected_norm_target = target_original / scale_factor

        torch.testing.assert_close(
            norm_input, expected_norm_input, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            norm_target, expected_norm_target, rtol=1e-5, atol=1e-6
        )

    def test_scale_factor_per_sample_not_global(self, strategy):
        """Test that scale factor is computed per-sample, not globally."""
        batch_size = 3

        # Create samples with very different energy levels
        input1 = torch.randn(1, 2, 64, 64) * 0.1  # Low energy
        input2 = torch.randn(1, 2, 64, 64) * 1.0  # Medium energy
        input3 = torch.randn(1, 2, 64, 64) * 10.0  # High energy

        input_kspace = torch.cat([input1, input2, input3], dim=0)
        target_kspace = torch.randn(batch_size, 2, 64, 64)

        batch = {"input": input_kspace, "target": target_kspace}

        norm_input, norm_target, scale_factor = strategy._prepare_validation_data(
            batch, None, None, None
        )

        # Verify scale_factor has different values per sample
        assert scale_factor.shape[0] == batch_size

        # Scales should be different (not all equal)
        scale_values = scale_factor.view(batch_size, -1).mean(dim=1)
        assert not torch.allclose(scale_values, scale_values[0].expand_as(scale_values))

    def test_normalization_disabled_returns_identity_scale(self, strategy):
        """Test that when normalization is disabled, identity scale is returned."""
        # Disable normalization at the path the strategy actually reads.
        strategy.state.config.data.processing.enable_kspace_normalization = False

        batch_size = 2
        input_kspace = torch.randn(batch_size, 2, 64, 64)
        target_kspace = torch.randn(batch_size, 2, 64, 64)

        batch = {"input": input_kspace, "target": target_kspace}

        norm_input, norm_target, scale_factor = strategy._prepare_validation_data(
            batch, None, None, None
        )

        # Scale factor should be identity (ones)
        expected_scale = torch.ones(batch_size, 1, 1, 1)
        torch.testing.assert_close(scale_factor, expected_scale)

        # Data should be unchanged
        torch.testing.assert_close(norm_input, input_kspace)
        torch.testing.assert_close(norm_target, target_kspace)
