"""Benchmarks for End-to-End Training Loop (Phase 6)."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.training.strategies.gan import GANTrainingStrategy


@pytest.fixture
def mock_strategy(benchmark_config):
    """Setup a full training strategy with mocked data loader."""
    device = benchmark_config["device"]

    # Minimal config
    config = MagicMock()
    config.training.training_mode = "gan"
    config.optimization.precision.enabled = True
    config.optimization.optimizer.learning_rate = 1e-4

    # Mock environment
    env = MagicMock()
    env.device = device
    env.generator = nn.Conv2d(1, 1, 3, padding=1).to(device)
    env.discriminator = nn.Conv2d(1, 1, 3, padding=1).to(device)
    env.opt_g = torch.optim.Adam(env.generator.parameters())
    env.opt_d = torch.optim.Adam(env.discriminator.parameters())

    strategy = GANTrainingStrategy(env=env, state=MagicMock())
    return strategy


def benchmark_training_iteration(benchmark, mock_strategy, benchmark_config):
    """Measure full training step throughput."""
    B, C, H, W = benchmark_config["batch_size"], 1, 128, 128
    device = benchmark_config["device"]

    batch = {
        "input": torch.randn(B, C, H, W, device=device),
        "target": torch.randn(B, C, H, W, device=device),
    }

    def _step():
        return mock_strategy.train_step(batch, epoch=0)

    # Warmup
    _step()

    benchmark(_step)
