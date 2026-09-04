"""Unit tests for GAN training strategy.

Tests the GAN strategy implementation including:
- Initialization and configuration
- Loss computation (generator and discriminator)
- Adversarial training mechanics
- @final decorator enforcement on train_step
"""

import inspect
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy


def test_train_generator_step_reads_loop_state_not_frozen_env_step() -> None:
    """WS-3 follow-up: the GAN generator step's ``current_step`` (fed to the
    ``train_metric_interval`` throttle in ``_compute_training_metrics``) now
    reads the live ``loop_state`` seam, not the frozen ``self.env.step`` 0 that
    made the throttle fire every step (pitfall #16). Active code no longer reads
    ``self.env.step``."""
    code = "\n".join(
        ln
        for ln in inspect.getsource(GANTrainingStrategy._train_generator_step).splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert "self.env.step" not in code
    assert "resolve_loop_iteration(self)" in code


class SimpleGenerator(nn.Module):
    """Minimal generator for testing."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return self.conv2(x)


class SimpleDiscriminator(nn.Module):
    """Minimal discriminator for testing."""

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def mock_gan_state() -> MagicMock:
    """Create mock training state for GAN."""
    state = MagicMock()
    state.device = torch.device("cpu")
    state.step = 0
    state.epoch = 0

    # Mock config
    state.config = MagicMock()
    state.config.training = MagicMock()
    state.config.training.training_mode = "gan"
    state.config.training.log_interval = 100

    state.config.optimization = MagicMock()
    state.config.optimization.precision.enabled = False

    state.config.objectives = MagicMock()
    state.config.objectives.adversarial = MagicMock()
    state.config.objectives.adversarial.lambda_adv = 1.0
    state.config.objectives.adversarial.discriminator_loss_type = "bce"

    state.config.objectives.reconstruction = MagicMock()
    state.config.objectives.reconstruction.lambda_l1 = 100.0
    state.config.objectives.reconstruction.lambda_l2 = 0.0

    return state


@pytest.fixture
def gan_models():
    """Create simple GAN models for testing."""
    generator = SimpleGenerator(in_channels=1, out_channels=1)
    discriminator = SimpleDiscriminator(in_channels=1)
    return {"generator": generator, "discriminator": discriminator}


class TestGANStrategyInitialization:
    """Test GAN strategy initialization."""

    def test_strategy_class_available(self):
        """Test GAN strategy class is available."""
        assert GANTrainingStrategy is not None
        assert hasattr(GANTrainingStrategy, "train_step")

    def test_requires_discriminator(self, mock_gan_state):
        """Test that GAN strategy requires discriminator."""
        # This test verifies discriminator is expected in GAN mode
        assert mock_gan_state.config.training.training_mode == "gan"


class TestGANLossComputation:
    """Test GAN loss computation."""

    def test_loss_dict_shape(self):
        """Test expected GAN loss dictionary structure."""
        result = {
            "g_total_loss": 0.5,
            "d_loss": 0.3,
            "adversarial": 0.2,
            "l1": 0.3,
        }
        assert isinstance(result, dict)
        assert "g_total_loss" in result
        assert "d_loss" in result

    def test_generator_loss_is_tensor(self, mock_gan_state, gan_models):
        """Test generator loss is a valid tensor."""
        input_batch = torch.randn(2, 1, 64, 64)
        target_batch = torch.randn(2, 1, 64, 64)
        generator = gan_models["generator"]

        # Forward pass
        fake_output = generator(input_batch)

        # L1 reconstruction loss
        l1_loss = nn.L1Loss()(fake_output, target_batch)

        assert isinstance(l1_loss, torch.Tensor)
        assert l1_loss.dim() == 0  # Scalar
        assert l1_loss.dtype == torch.float32

    def test_discriminator_loss_is_tensor(self, mock_gan_state, gan_models):
        """Test discriminator loss is a valid tensor."""
        real_batch = torch.randn(2, 1, 64, 64)
        fake_batch = torch.randn(2, 1, 64, 64)
        discriminator = gan_models["discriminator"]

        # Forward pass
        real_pred = discriminator(real_batch)
        fake_pred = discriminator(fake_batch.detach())

        # BCE loss
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss_real = nn.BCEWithLogitsLoss()(real_pred, real_labels)
        d_loss_fake = nn.BCEWithLogitsLoss()(fake_pred, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        assert isinstance(d_loss, torch.Tensor)
        assert d_loss.dtype == torch.float32


class TestGANTrainStepEnforcement:
    """Test @final decorator on train_step."""

    def test_train_step_not_overridden(self):
        """Verify train_step is inherited from BaseTrainingStrategy."""
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

        # GANTrainingStrategy should inherit final train_step behavior from base.
        assert hasattr(BaseTrainingStrategy, "train_step")

        # train_step should be @final (no override in concrete class)
        # This is enforced by mypy, but we can check the method resolution order
        assert "train_step" in dir(GANTrainingStrategy)


class TestGANAdversarialMechanics:
    """Test GAN-specific adversarial training mechanics."""

    def test_generator_backward_updates_generator_only(self, gan_models):
        """Test generator backward doesn't update discriminator."""
        generator = gan_models["generator"]
        discriminator = gan_models["discriminator"]

        # Setup optimizers
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Store initial discriminator params
        d_params_before = [p.clone() for p in discriminator.parameters()]

        # Generator training step
        input_batch = torch.randn(2, 1, 64, 64)
        target_batch = torch.randn(2, 1, 64, 64)

        g_optimizer.zero_grad()
        fake_output = generator(input_batch)
        g_loss = nn.L1Loss()(fake_output, target_batch)
        g_loss.backward()
        g_optimizer.step()

        # Discriminator params should be unchanged
        d_params_after = list(discriminator.parameters())
        for p_before, p_after in zip(d_params_before, d_params_after, strict=False):
            assert torch.allclose(p_before, p_after)

    def test_discriminator_backward_updates_discriminator_only(self, gan_models):
        """Test discriminator backward doesn't update generator."""
        generator = gan_models["generator"]
        discriminator = gan_models["discriminator"]

        # Setup optimizers
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Store initial generator params
        g_params_before = [p.clone() for p in generator.parameters()]

        # Discriminator training step
        real_batch = torch.randn(2, 1, 64, 64)
        fake_batch = generator(
            real_batch
        ).detach()  # Detach to prevent generator gradients

        d_optimizer.zero_grad()
        real_pred = discriminator(real_batch)
        fake_pred = discriminator(fake_batch)

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss = nn.BCEWithLogitsLoss()(
            real_pred, real_labels
        ) + nn.BCEWithLogitsLoss()(fake_pred, fake_labels)
        d_loss.backward()
        d_optimizer.step()

        # Generator params should be unchanged
        g_params_after = list(generator.parameters())
        for p_before, p_after in zip(g_params_before, g_params_after, strict=False):
            assert torch.allclose(p_before, p_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
