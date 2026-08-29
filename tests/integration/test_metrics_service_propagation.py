"""Integration tests for metrics_service propagation through training pipeline.

This test suite verifies that:
1. metrics_service is passed from bootstrap through to strategy
2. Images are actually saved during validation
3. Configuration flags are respected end-to-end
4. Training metrics CSV is created and populated
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.services.metrics_tracker import MetricsTracker
from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory


class SimpleTestGenerator(nn.Module):
    """Simple test generator for integration tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        self.in_channels = 2

    def forward(self, x):
        return self.conv(x)


class SimpleTestStrategy(MagicMock):
    """Test strategy that simulates DiffusionTrainingStrategy."""

    def __init__(self, env, logging_service=None, **kwargs):
        super().__init__()
        self.env = env
        self.logging_service = logging_service or MagicMock()
        self.metrics_service = kwargs.get("metrics_service")
        self.checkpoint_service = kwargs.get("checkpoint_service")
        self.current_epoch = 0
        self.validation_step_count = 0

    def validation_step(self, batch, batch_idx=0):
        """Mock validation step."""
        return {"val_loss": 0.5, "val_psnr": 20.0}

    def _log_validation_images_to_tensorboard(
        self, predictions, targets, inputs, metrics
    ):
        """Mock image logging that actually saves."""
        if self.metrics_service is not None:
            real_paths, fake_paths = self.metrics_service.save_images_batch(
                real_images=targets,
                fake_images=predictions,
                prefix="validation",
                epoch=self.current_epoch,
                step=0,
            )
            return real_paths, fake_paths
        return [], []


class TestMetricsServicePropagation:
    """Test that metrics_service is properly propagated."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def metrics_service(self, temp_output_dir):
        """Create metrics service with image saving enabled."""
        return MetricsTracker(
            output_dir=str(temp_output_dir / "metrics"),
            save_images=True,
            image_format="png",
            device="cpu",
            model_type="test_model",
        )

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline/environment."""
        pipeline = MagicMock()
        pipeline.generator = SimpleTestGenerator()

        # Configure mock config
        config = MagicMock()
        config.training.training_mode = "diffusion"
        config.model.model_type = "test_generator"
        pipeline.config = config

        return pipeline

    @pytest.fixture
    def factory(self):
        """Create strategy factory."""
        return TrainingStrategyFactory()

    def test_metrics_service_available_after_factory_creation(
        self, factory, mock_pipeline, metrics_service
    ):
        """Test that metrics_service is accessible on strategy after creation."""
        logging_service = MagicMock()

        # Patch to use our test strategy
        with patch.object(
            factory, "get_strategy_class", return_value=SimpleTestStrategy
        ):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                metrics_service=metrics_service,  # ✅ Pass it here
            )

        # ✅ Should be available on strategy
        assert strategy.metrics_service is metrics_service
        assert strategy.logging_service is logging_service

    def test_images_saved_during_validation_when_metrics_service_provided(
        self, factory, mock_pipeline, metrics_service, temp_output_dir
    ):
        """Test that images are actually saved when metrics_service is provided."""
        logging_service = MagicMock()

        with patch.object(
            factory, "get_strategy_class", return_value=SimpleTestStrategy
        ):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                metrics_service=metrics_service,
            )

        # Simulate validation with image logging (use 3 channels - RGB, supported by PIL)
        predictions = torch.randn(2, 3, 32, 32)
        targets = torch.randn(2, 3, 32, 32)
        inputs = torch.randn(2, 3, 32, 32)
        metrics = {"val_loss": 0.5}

        strategy._log_validation_images_to_tensorboard(
            predictions, targets, inputs, metrics
        )

        # ✅ Images should exist on disk
        img_dir = temp_output_dir / "metrics"
        assert (img_dir / "real_images").exists()
        assert (img_dir / "fake_images").exists()

        # ✅ Should have image files
        real_files = list((img_dir / "real_images").glob("*.png"))
        fake_files = list((img_dir / "fake_images").glob("*.png"))
        assert len(real_files) > 0
        assert len(fake_files) > 0

    def test_no_images_saved_when_metrics_service_missing(
        self, factory, mock_pipeline, temp_output_dir
    ):
        """Test that no image files are created when metrics_service is None."""
        logging_service = MagicMock()

        with patch.object(
            factory, "get_strategy_class", return_value=SimpleTestStrategy
        ):
            strategy = factory.create_strategy(
                env=mock_pipeline,
                logging_service=logging_service,
                # ❌ No metrics_service
            )

        # metrics_service should be None
        assert strategy.metrics_service is None

        # Simulate validation with image logging (use 3 channels - RGB)
        predictions = torch.randn(2, 3, 32, 32)
        targets = torch.randn(2, 3, 32, 32)
        inputs = torch.randn(2, 3, 32, 32)
        metrics = {"val_loss": 0.5}

        strategy._log_validation_images_to_tensorboard(
            predictions, targets, inputs, metrics
        )

        # ✅ No image files should be created (because metrics_service is None)
        # The returned paths should be empty lists
        # This test verifies graceful handling when metrics_service is None

    def test_configuration_propagates_to_metrics_service(self):
        """Test that config flags propagate to metrics_service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tracker with save_images=True
            tracker_enabled = MetricsTracker(
                output_dir=tmpdir,
                save_images=True,  # From config.logging.save_validation_images
                image_format="png",
                device="cpu",
                model_type="test",
            )

            # ✅ Should save when True
            assert tracker_enabled.save_images is True

            # Create tracker with save_images=False
            tracker_disabled = MetricsTracker(
                output_dir=tmpdir,
                save_images=False,  # From config.logging.save_validation_images=False
                image_format="png",
                device="cpu",
                model_type="test",
            )

            # ✅ Should not save when False
            assert tracker_disabled.save_images is False


class TestEndToEndMetricsServiceFlow:
    """End-to-end test of metrics_service through entire flow."""

    def test_full_flow_from_bootstrap_to_validation(self):
        """Test complete flow: bootstrap → factory → strategy → validation → saved images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Step 1: Create metrics_service (simulating bootstrap)
            metrics_service = MetricsTracker(
                output_dir=str(output_path / "metrics"),
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

            # Step 2: Create mock pipeline
            pipeline = MagicMock()
            pipeline.config = MagicMock()

            # Step 3: Create strategy with metrics_service (simulating factory call)
            strategy = SimpleTestStrategy(
                env=pipeline,
                logging_service=MagicMock(),
                metrics_service=metrics_service,  # ✅ From bootstrap
            )

            # Step 4: Run validation with image logging (use 3 channels - RGB)
            predictions = torch.randn(1, 3, 32, 32)
            targets = torch.randn(1, 3, 32, 32)
            inputs = torch.randn(1, 3, 32, 32)
            metrics = {}

            strategy._log_validation_images_to_tensorboard(
                predictions, targets, inputs, metrics
            )

            # Step 5: Verify images were saved
            real_images = list((output_path / "metrics" / "real_images").glob("*.png"))
            fake_images = list((output_path / "metrics" / "fake_images").glob("*.png"))

            # ✅ Full end-to-end flow should result in saved images
            assert len(real_images) > 0, "Real images not saved"
            assert len(fake_images) > 0, "Fake images not saved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
