"""Training resume integrity test utilities."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.infrastructure.services.checkpoint_service import CheckpointService
from spectramr.infrastructure.training.strategies import GANTrainingStrategy
from spectramr.models.factories.model_factory import get_model_factory
from spectramr.shared.utils.deterministic_seed import create_deterministic_harness


class TrainingResumeTester:
    """Tests training resume integrity by comparing checkpoint continuation."""

    def __init__(self, temp_dir: str | None = None):
        """Initialize the training resume tester.

        Args:
            temp_dir: Temporary directory for test files. Auto-created if None.

        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        config = CheckpointConfigSchema(
            checkpoint_dir=str(self.temp_dir / "checkpoints"),
        )
        self.checkpoint_service = CheckpointService(config=config)

        # Setup deterministic seeding
        self.seed_harness = create_deterministic_harness()

    def create_test_model_and_data(self, model_type: str = "standard_unet"):
        """Create test model and synthetic data for training."""
        factory = get_model_factory()

        # Create generator
        generator = factory.create_generator(model_type, in_channels=1, out_channels=1)

        # Create discriminator
        discriminator = factory.create_discriminator("patchgan", in_channels=1)

        # Create synthetic data
        batch_size = 4
        input_shape = (batch_size, 1, 64, 64)

        lr_batch = torch.randn(*input_shape)  # Low resolution input
        hr_batch = torch.randn(*input_shape)  # High resolution target

        return generator, discriminator, lr_batch, hr_batch

    def _create_mock_config(self):
        """Create a mock training configuration."""
        from unittest.mock import MagicMock

        config = MagicMock()
        config.training = MagicMock()
        config.training.training_mode = "gan"
        config.model = MagicMock()
        config.model.model_type = "standard_unet"
        config.optimization = MagicMock()
        config.optimization.optimizer.learning_rate = 1e-4
        config.optimization.precision.enabled = False
        config.objectives = MagicMock()
        config.objectives.reconstruction = MagicMock()
        config.objectives.reconstruction.lambda_l1 = 1.0
        config.objectives.gan = MagicMock()
        config.objectives.gan.lambda_adversarial = 1.0
        return config

    def run_training_epochs(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        lr_batch: torch.Tensor,
        hr_batch: torch.Tensor,
        num_epochs: int,
        save_checkpoints: bool = True,
    ) -> list[dict[str, Any]]:
        """Run training for specified number of epochs.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            lr_batch: Low resolution batch
            hr_batch: High resolution batch
            num_epochs: Number of epochs to train
            save_checkpoints: Whether to save checkpoints

        Returns:
            list of metrics for each epoch

        """
        # Setup optimizers
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Setup loss functions
        g_criterion = nn.MSELoss()
        d_criterion = nn.BCEWithLogitsLoss()

        # Create training environment using mock helper
        training_env = create_mock_training_env(
            config=self._create_mock_config(),
            device="cpu",
            generator=generator,
            discriminator=discriminator,
            opt_g=g_optimizer,
            opt_d=d_optimizer,
            model_type="gan",
        )

        # Create training strategy
        strategy = GANTrainingStrategy(env=training_env)

        metrics_history = []

        for epoch in range(num_epochs):
            # Train for one step (simulating one epoch)
            losses = strategy.train_step(
                batch=lr_batch, epoch=epoch, lr_batch=lr_batch, hr_batch=hr_batch
            )

            metrics = {
                "epoch": epoch,
                "g_loss": losses.get("g_loss", 0.0),
                "d_loss": losses.get("d_loss", 0.0),
                "total_loss": losses.get("total_loss", 0.0),
            }

            metrics_history.append(metrics)

            # Save checkpoint if requested
            if save_checkpoints:
                checkpoint_path = self.checkpoint_service.save_checkpoint(
                    model=generator,
                    optimizer=g_optimizer,
                    epoch=epoch,
                    loss=metrics["total_loss"],
                    discriminator_state_dict=discriminator.state_dict(),
                    d_optimizer_state_dict=d_optimizer.state_dict(),
                    checkpoint_name=f"epoch_{epoch}",
                )
                metrics["checkpoint_path"] = checkpoint_path

            self.logger.info(f"Epoch {epoch}: {metrics}")

        return metrics_history

    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        additional_epochs: int = 2,
    ) -> list[dict[str, Any]]:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            additional_epochs: Number of additional epochs to train

        Returns:
            list of metrics for resumed training

        """
        # Load checkpoint
        checkpoint_data = self.checkpoint_service.load_checkpoint(checkpoint_path)

        # Recreate models and optimizers
        factory = get_model_factory()
        generator = factory.create_generator(
            "standard_unet",
            in_channels=1,
            out_channels=1,
        )
        discriminator = factory.create_discriminator("patchgan", in_channels=1)

        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Load states
        generator.load_state_dict(checkpoint_data["model_state_dict"])
        g_optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        if "discriminator_state_dict" in checkpoint_data:
            disc_state = checkpoint_data["discriminator_state_dict"]
            discriminator.load_state_dict(disc_state)
        if "d_optimizer_state_dict" in checkpoint_data:
            d_opt_state = checkpoint_data["d_optimizer_state_dict"]
            d_optimizer.load_state_dict(d_opt_state)

        # Create synthetic data (same as original)
        batch_size = 4
        input_shape = (batch_size, 1, 64, 64)
        lr_batch = torch.randn(*input_shape)
        hr_batch = torch.randn(*input_shape)

        # Continue training
        start_epoch = checkpoint_data.get("epoch", 0) + 1
        resumed_metrics = self.run_training_epochs(
            generator,
            discriminator,
            lr_batch,
            hr_batch,
            additional_epochs,
            save_checkpoints=False,
        )

        # Adjust epoch numbers
        for metrics in resumed_metrics:
            metrics["epoch"] += start_epoch

        return resumed_metrics

    def test_resume_integrity(
        self,
        model_type: str = "standard_unet",
        initial_epochs: int = 5,
        resume_epochs: int = 2,
        tolerance: float = 1e-6,
    ) -> dict[str, Any]:
        """Test training resume integrity.

        Args:
            model_type: Type of model to test
            initial_epochs: Number of initial training epochs
            resume_epochs: Number of epochs to train after resume
            tolerance: Tolerance for metrics comparison

        Returns:
            Dictionary containing test results

        """
        self.logger.info("Starting training resume integrity test...")

        # Set deterministic seed
        test_config = {
            "model_type": model_type,
            "initial_epochs": initial_epochs,
            "resume_epochs": resume_epochs,
            "batch_size": 4,
            "learning_rate": 1e-4,
        }
        self.seed_harness.apply_deterministic_seed(test_config)

        # Create models and data
        generator, discriminator, lr_batch, hr_batch = self.create_test_model_and_data(
            model_type,
        )

        # Run initial training
        msg = f"Running initial training for {initial_epochs} epochs..."
        self.logger.info(msg)
        initial_metrics = self.run_training_epochs(
            generator,
            discriminator,
            lr_batch,
            hr_batch,
            initial_epochs,
        )

        # Get checkpoint from last epoch
        last_checkpoint = initial_metrics[-1]["checkpoint_path"]

        # Resume training
        msg = f"Resuming training for {resume_epochs} additional epochs..."
        self.logger.info(msg)
        resumed_metrics = self.resume_from_checkpoint(last_checkpoint, resume_epochs)

        # Run fresh training for comparison
        self.logger.info("Running fresh training for comparison...")
        fresh_generator, fresh_discriminator, _, _ = self.create_test_model_and_data(
            model_type,
        )
        fresh_metrics = self.run_training_epochs(
            fresh_generator,
            fresh_discriminator,
            lr_batch,
            hr_batch,
            initial_epochs + resume_epochs,
            save_checkpoints=False,
        )

        # Compare metrics
        comparison_results = self._compare_metrics(
            initial_metrics + resumed_metrics,
            fresh_metrics,
            tolerance,
        )

        # Prepare results
        results = {
            "test_config": test_config,
            "initial_metrics": initial_metrics,
            "resumed_metrics": resumed_metrics,
            "fresh_metrics": fresh_metrics,
            "comparison": comparison_results,
            "integrity_passed": comparison_results["all_within_tolerance"],
            "max_difference": comparison_results["max_difference"],
            "checkpoint_path": last_checkpoint,
        }

        passed = results["integrity_passed"]
        self.logger.info(f"Resume integrity test completed. Passed: {passed}")
        return results

    def _compare_metrics(
        self,
        resumed_metrics: list[dict[str, Any]],
        fresh_metrics: list[dict[str, Any]],
        tolerance: float,
    ) -> dict[str, Any]:
        """Compare metrics between resumed and fresh training."""
        if len(resumed_metrics) != len(fresh_metrics):
            return {
                "all_within_tolerance": False,
                "max_difference": float("inf"),
                "differences": [],
                "error": "Different number of epochs",
            }

        differences = []
        max_diff = 0.0

        for i, (resumed, fresh) in enumerate(
            zip(resumed_metrics, fresh_metrics, strict=False)
        ):
            epoch_diff = abs(resumed["total_loss"] - fresh["total_loss"])
            differences.append(
                {
                    "epoch": i,
                    "resumed_loss": resumed["total_loss"],
                    "fresh_loss": fresh["total_loss"],
                    "difference": epoch_diff,
                },
            )
            max_diff = max(max_diff, epoch_diff)

        all_within_tolerance = max_diff <= tolerance

        return {
            "all_within_tolerance": all_within_tolerance,
            "max_difference": max_diff,
            "differences": differences,
        }

    def save_test_results(self, results: dict[str, Any], output_path: str) -> None:
        """Save test results to file.

        Args:
            results: Test results dictionary

        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors and non-serializable objects to strings
        serializable_results = self._make_serializable(results)

        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Test results saved to: {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable by converting tensors and paths."""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            msg = f"Cleaned up temporary directory: {self.temp_dir}"
            self.logger.info(msg)


def run_resume_integrity_test(
    model_type: str = "standard_unet",
    initial_epochs: int = 5,
    resume_epochs: int = 2,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run a training resume integrity test.

    Args:
        model_type: Type of model to test
        initial_epochs: Number of initial training epochs
        resume_epochs: Number of epochs to train after resume
        output_path: Path to save results (optional)

    Returns:
        Dictionary containing test results

    """
    tester = TrainingResumeTester()

    try:
        results = tester.test_resume_integrity(
            model_type=model_type,
            initial_epochs=initial_epochs,
            resume_epochs=resume_epochs,
        )

        if output_path:
            tester.save_test_results(results, output_path)

        return results

    finally:
        tester.cleanup()
