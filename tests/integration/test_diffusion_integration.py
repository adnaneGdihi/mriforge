"""Integration tests for Diffusion training pipeline."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy


@pytest.mark.integration
@pytest.mark.skip(
    reason="Legacy test harness issues with MagicMock configuration causing TypeErrors"
)
def test_diffusion_pipeline_execution():
    """Test full diffusion pipeline execution."""
    device = torch.device("cpu")

    # 1. Models
    model = nn.Conv2d(1, 1, 3, padding=1).to(device)  # Simple denoiser

    # 2. Env
    env = MagicMock()
    env.device = device
    env.generator = model
    env.opt_g = torch.optim.Adam(model.parameters())

    # 3. Strategy
    state = MagicMock()
    state.device = device
    state.epoch = 0
    state.config.training.training_mode = "diffusion"
    state.config.training.diffusion.timesteps = 10  # Short schedule
    # Disable reconstruction losses to avoid builder error
    state.config.losses.reconstruction = None
    state.config.losses.physics = None
    state.config.deep_supervision_weight = 0
    state.config.losses.gan = None  # Disable GAN losses (R1)
    env.config = state.config

    # Patch dependencies
    # DiffusionScheduler is used in DiffusionStrategyMixin
    with patch(
        "mriforge.infrastructure.training.utils.diffusion_mixin.DiffusionScheduler"
    ) as MockSched:
        # Setup scheduler mock
        scheduler = MockSched.return_value
        scheduler.sample_timesteps.return_value = torch.tensor([0, 1], device=device)
        scheduler.add_noise.return_value = (
            torch.randn(2, 1, 16, 16),
            torch.randn(2, 1, 16, 16),
        )  # noisy, noise

        strategy = DiffusionTrainingStrategy(env=env, state=state)

        # 4. Loop
        for i in range(2):
            batch = {
                "input": torch.randn(2, 1, 16, 16).to(device),
                "target": torch.randn(2, 1, 16, 16).to(device),
            }
            # Default diffusion strategy expects target (HR) and maybe condition
            metrics = strategy.train_step(batch, epoch=0)

            assert metrics is not None
            assert "g_total_loss" in metrics
