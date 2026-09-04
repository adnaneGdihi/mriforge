"""
Phase 6: Training Fixtures for Integration Tests
Provides reusable test utilities and fixtures.
"""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.loss import (
    L1LossConfig,
    LossConfigSchema,
    ReconstructionLossesConfig,
    SSIMLossConfig,
)
from spectramr.infrastructure.di.di_container import DIContainer
from spectramr.infrastructure.loss_audit import verify_startup_losses
from spectramr.infrastructure.training.builders.optimization_builder import (
    OptimizationBuilder,
)
from spectramr.models.losses.composed_loss import ComposedLoss, WeightedLoss
from spectramr.models.losses.registry import create_loss


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    loss: float
    metrics: dict[str, float]
    elapsed_time: float
    batch_size: int


@dataclass
class TrainingConfig:
    """Configuration for training fixture."""

    batch_size: int = 4
    height: int = 64
    width: int = 64
    num_steps: int = 5
    learning_rate: float = 1e-3
    device: str = "cpu"


class MockModel(nn.Module):
    """Mock U-Net model for testing."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """Initialize mock model.

        Args:
            in_channels: Input channels
            out_channels: Output channels
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        out = self.decoder(bottleneck)
        return torch.tanh(out)


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing."""

    def __init__(
        self,
        num_samples: int = 32,
        height: int = 64,
        width: int = 64,
        channels: int = 3,
    ):
        """Initialize mock dataset.

        Args:
            num_samples: Number of samples
            height: Image height
            width: Image width
            channels: Number of channels
        """
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.channels = channels

    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input' and 'target' tensors
        """
        return {
            "input": torch.randn(self.channels, self.height, self.width),
            "target": torch.randn(self.channels, self.height, self.width),
        }


class TrainingFixture:
    """Fixture for integration testing of training pipelines."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        loss_config: LossConfigSchema | None = None,
    ):
        """Initialize training fixture.

        Args:
            config: Training configuration
            loss_config: Loss configuration
        """
        self.config = config or TrainingConfig()
        self.loss_config = loss_config or self._create_default_loss_config()
        self.device = torch.device(self.config.device)

    def _create_default_loss_config(self) -> LossConfigSchema:
        """Create default loss configuration.

        Returns:
            LossConfigSchema with L1 + SSIM
        """
        return LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                l1=L1LossConfig(enabled=True, weight=10.0),
                ssim=SSIMLossConfig(enabled=True, weight=0.1),
            )
        )

    def setup(self) -> tuple[MockModel, nn.Module, torch.optim.Optimizer, DIContainer]:
        """Setup complete training infrastructure.

        Returns:
            (model, loss, optimizer, container)

        Raises:
            RuntimeError: If configuration is invalid
        """
        # Validate loss configuration
        verify_startup_losses(self.loss_config)

        # Create model
        model = MockModel().to(self.device)

        # Create loss
        # Use ComposedLoss (Phase 5 implementation with metrics support)
        weighted_losses = []
        if self.loss_config.reconstruction.l1.enabled:
            weighted_losses.append(
                WeightedLoss(
                    name="l1",
                    loss_fn=create_loss("l1"),
                    weight=self.loss_config.reconstruction.l1.weight,
                )
            )
        if self.loss_config.reconstruction.ssim.enabled:
            weighted_losses.append(
                WeightedLoss(
                    name="ssim",
                    loss_fn=create_loss("ssim"),
                    weight=self.loss_config.reconstruction.ssim.weight,
                )
            )

        loss = ComposedLoss(weighted_losses).to(self.device)

        # Create optimizer
        optimizer = OptimizationBuilder.create_single_optimizer(
            model.parameters(),
            learning_rate=self.config.learning_rate,
            optimizer_type="adam",
        )

        # Build DI container
        # Note: Would need proper config object
        container = None  # DIContainer not fully mocked

        return model, loss, optimizer, container

    def create_batch(self) -> dict[str, torch.Tensor]:
        """Create a random training batch.

        Returns:
            Dictionary with 'input' and 'target' tensors
        """
        return {
            "input": torch.randn(
                self.config.batch_size, 3, self.config.height, self.config.width
            ).to(self.device),
            "target": torch.randn(
                self.config.batch_size, 3, self.config.height, self.config.width
            ).to(self.device),
        }

    def create_dataloader(
        self, num_samples: int = 32, batch_size: int | None = None
    ) -> torch.utils.data.DataLoader:
        """Create a mock data loader.

        Args:
            num_samples: Number of samples in dataset
            batch_size: Batch size (defaults to config.batch_size)

        Returns:
            DataLoader with mock data
        """
        batch_size = batch_size or self.config.batch_size
        dataset = MockDataset(
            num_samples=num_samples,
            height=self.config.height,
            width=self.config.width,
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def training_step(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: dict[str, torch.Tensor],
    ) -> TrainingMetrics:
        """Execute a single training step.

        Args:
            model: Model to train
            loss: Loss function
            optimizer: Optimizer
            batch: Training batch

        Returns:
            TrainingMetrics with loss and metrics
        """
        start_time = time.time()

        # Forward pass
        pred = model(batch["input"])
        target = batch["target"]

        # Loss and metrics
        loss_val, metrics = loss(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        elapsed = time.time() - start_time

        return TrainingMetrics(
            loss=loss_val.item(),
            metrics=metrics,
            elapsed_time=elapsed,
            batch_size=self.config.batch_size,
        )

    def validation_epoch(
        self,
        model: nn.Module,
        loss: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Run validation epoch.

        Args:
            model: Model to validate
            loss: Loss function
            dataloader: Validation dataloader

        Returns:
            Aggregated metrics dictionary
        """
        model.eval()

        total_loss = 0.0
        metric_sums: dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                pred = model(batch["input"])
                target = batch["target"]

                loss_val, metrics = loss(pred, target)

                total_loss += loss_val.item()

                for key, value in metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value

                num_batches += 1

        # Average
        result = {"loss": total_loss / num_batches}
        for key, value in metric_sums.items():
            result[key] = value / num_batches

        model.train()
        return result

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> Path:
        """Save checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Metrics to save

        Returns:
            Path to checkpoint
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / f"checkpoint_epoch_{epoch}.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                checkpoint_path,
            )

            return checkpoint_path

    def load_checkpoint(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: Path
    ) -> tuple[int, dict[str, float]]:
        """Load checkpoint.

        Args:
            model: Model to load into
            optimizer: Optimizer to load into
            checkpoint_path: Path to checkpoint

        Returns:
            (epoch, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["epoch"], checkpoint["metrics"]


# Pytest fixtures


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create training configuration."""
    return TrainingConfig(batch_size=4, height=64, width=64, num_steps=5)


@pytest.fixture
def loss_config() -> LossConfigSchema:
    """Create loss configuration."""
    return LossConfigSchema(
        reconstruction=ReconstructionLossesConfig(
            l1=L1LossConfig(enabled=True, weight=10.0),
            ssim=SSIMLossConfig(enabled=True, weight=0.1),
        )
    )


@pytest.fixture
def training_fixture(
    training_config: TrainingConfig, loss_config: LossConfigSchema
) -> TrainingFixture:
    """Create training fixture."""
    return TrainingFixture(config=training_config, loss_config=loss_config)


@pytest.fixture
def mock_model() -> MockModel:
    """Create mock model."""
    return MockModel()


@pytest.fixture
def mock_dataloader() -> torch.utils.data.DataLoader:
    """Create mock dataloader."""
    dataset = MockDataset(num_samples=32, height=64, width=64)
    return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
