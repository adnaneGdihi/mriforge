"""Integration tests for Reconstruction training pipeline."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)


@pytest.mark.integration
def test_reconstruction_pipeline_execution():
    """Test full reconstruction pipeline execution."""
    device = torch.device("cpu")

    # 1. Models
    model = nn.Conv2d(1, 1, 3, padding=1).to(device)

    # 2. Env
    env = MagicMock()
    env.device = device
    env.generator = model
    env.opt_g = torch.optim.Adam(model.parameters())
    env.losses = {"l1": nn.L1Loss()}  # Explicit losses in env

    # 3. Strategy
    state = MagicMock()
    state.device = device
    state.epoch = 0
    state.config.training.training_mode = "reconstruction"

    # Fix: Setup nested config for loss builder
    # loss_builder.py checks: if getattr(recon_config, "enable_l1", False) and recon_config.lambda_l1 > 0:
    # Spec with the real schema class so misspelled v6 paths fail loud.
    from spectramr.config.schemas.loss import ReconstructionLossesConfig

    recon_config = MagicMock(spec=ReconstructionLossesConfig)
    recon_config.enable_l1 = True
    recon_config.lambda_l1 = 1.0

    # Ensure nested access works: state.config.losses.reconstruction
    state.config.losses.reconstruction = recon_config

    # Disable other losses and physics sections to avoid MagicMock issues
    state.config.losses.physics = None
    state.config.losses.gan = None
    state.config.physics = None
    state.config.deep_supervision_weight = 0.0
    state.config.training.enforce_output_range = False
    state.config.use_mc_dropout = False
    state.config.metrics.enable_tracking = False

    # Optimization config
    state.config.optimization.gradient.clip.enabled = False
    state.config.optimization.gradient.clip.value = 1.0
    state.config.optimization.precision.enabled = False
    state.config.optimization.gradient.accumulation_steps = 1

    # Logging config
    state.config.logging.log_gradients = False
    state.config.logging.log_interval = 100

    # Model config
    state.config.model.target_domain = "image"

    # BaseTrainingStrategy prioritizes env.config if env is present
    env.config = state.config

    # Strategy setup
    strategy = ReconstructionTrainingStrategy(env=env, state=state)

    # 4. Loop
    for i in range(2):
        batch = {
            "input": torch.randn(2, 1, 16, 16).to(device),
            "target": torch.randn(2, 1, 16, 16).to(device),
        }
        metrics = strategy.train_step(batch, epoch=0)

        assert metrics is not None
        assert "g_total_loss" in metrics
        # Check if l1 loss is present (aggregated or individual)
        # Strategy might key it differently depending on mixin

        # Also verify gradients
        # We can check if optimizer step was called if we wrap it
        # But for integration, just running without error is good first step.
