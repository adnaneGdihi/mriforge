"""Integration tests for GAN training pipeline."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy


@pytest.fixture
def mock_gan_settings():
    """Create settings for GAN training."""
    settings = MagicMock()
    # Mock the configuration structure
    settings.training = MagicMock()
    settings.training.training_mode = "gan"
    settings.training.gan_config = MagicMock()
    settings.training.gan_config.lambda_adv = 1.0

    settings.data = MagicMock()
    settings.data.loader.batch_size = 2

    settings.optimization = MagicMock()
    settings.optimization.precision.enabled = False  # Must be bool for torch.autocast
    settings.optimization.gradient.clip.enabled = True
    settings.optimization.gradient.clip.value = 1.0

    settings.model = MagicMock()
    settings.model.generator.model_type = "unet"

    settings.physics = MagicMock()
    settings.physics.kspace.enable_kspace_recon = False

    return settings


@pytest.fixture
def mock_gan_env(mock_gan_settings):
    """Create environment with simple models."""
    env = MagicMock()
    env.device = torch.device("cpu")
    env.settings = mock_gan_settings
    env.config = mock_gan_settings  # Ensure config is available

    # Simple models
    env.generator = nn.Conv2d(1, 1, 3, padding=1)
    env.discriminator = nn.Conv2d(1, 1, 3, padding=1)

    # Optimizers
    env.opt_g = torch.optim.Adam(env.generator.parameters())
    env.opt_d = torch.optim.Adam(env.discriminator.parameters())

    return env


def test_gan_training_step_integration(mock_gan_env):
    """Verify GAN strategy runs a full training step without error."""
    # Setup strategy
    state = MagicMock()
    state.device = torch.device("cpu")
    state.config = mock_gan_env.settings

    strategy = GANTrainingStrategy(env=mock_gan_env, state=state)

    # Mock batch
    batch = {"input": torch.randn(2, 1, 32, 32), "target": torch.randn(2, 1, 32, 32)}

    # Run step. ``GANTrainingStrategy.train_step`` returns a list of step
    # descriptors (closures + optimizers + models) instead of a metrics dict —
    # the metrics are recorded on the strategy and exposed via ``get_last_metrics()``
    # after the closures fire.
    step_configs = strategy.train_step(batch, epoch=0)

    assert isinstance(step_configs, list) and step_configs
    names = {cfg["name"] for cfg in step_configs}
    assert "generator" in names
    assert "discriminator" in names
    for cfg in step_configs:
        assert callable(cfg["closure"])
        assert cfg["optimizer"] is not None
        assert cfg["model"] is not None


@pytest.mark.integration
def test_gan_pipeline_execution():
    """Test full pipeline execution with mocked dataloader."""
    # This represents a "mini" training loop
    device = torch.device("cpu")

    # 1. Models
    gen = nn.Conv2d(1, 1, 3, padding=1).to(device)
    disc = nn.Conv2d(1, 1, 3, padding=1).to(device)

    # 2. Env
    env = MagicMock()
    env.device = device
    env.generator = gen
    env.discriminator = disc
    env.opt_g = torch.optim.Adam(gen.parameters())
    env.opt_d = torch.optim.Adam(disc.parameters())

    # Setup config on env
    env.config = MagicMock()
    env.config.training = MagicMock()
    env.config.training.training_mode = "gan"
    env.config.training.gan_config = MagicMock()
    env.config.training.gan_config.lambda_adv = 1.0

    env.config.optimization = MagicMock()
    env.config.optimization.precision.enabled = False
    env.config.optimization.gradient.clip.enabled = False
    env.config.optimization.gradient.clip.value = 1.0

    env.config.physics = MagicMock()
    env.config.physics.kspace.enable_kspace_recon = False

    env.config.logging = MagicMock()
    env.config.logging.log_gradients = False

    # 3. Strategy (state is legacy but passed for compatibility if needed)
    state = MagicMock()
    state.device = device
    state.epoch = 0
    state.config = env.config

    # Patch loss computer or ensure defaults work
    with patch(
        "mriforge.infrastructure.training.strategies.mixins.adversarial.UnifiedGANLossComputer"
    ) as MockLoss:
        # Configure mock loss computer to return tensors
        mock_computer = MockLoss.return_value

        # Generator result
        g_result = MagicMock()
        g_result.total = torch.tensor(0.5, requires_grad=True)
        g_result.losses = {"g_loss": 0.5}
        mock_computer.compute_generator_loss.return_value = g_result

        # Discriminator result
        d_result = MagicMock()
        d_result.total = torch.tensor(0.5, requires_grad=True)
        d_result.losses = {"d_loss": 0.5}
        mock_computer.compute_discriminator_loss.return_value = d_result

        strategy = GANTrainingStrategy(env=env, state=state)

        # 4. Loop
        for i in range(2):
            batch = {
                "input": torch.randn(2, 1, 16, 16).to(device),
                "target": torch.randn(2, 1, 16, 16).to(device),
            }
            metrics = strategy.train_step(batch, epoch=0)

            assert metrics is not None
            assert "g_total_loss" in metrics
