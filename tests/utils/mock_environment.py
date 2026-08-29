from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.optim import Optimizer

from mriforge.infrastructure.training.builders.environment import TrainingEnvironment


def create_mock_training_env(
    config: Any,
    device: torch.device | str = "cpu",
    generator: nn.Module | None = None,
    discriminator: nn.Module | None = None,
    opt_g: Optimizer | None = None,
    opt_d: Optimizer | None = None,
    model_type: str = "reconstruction",
) -> Any:
    """Create a mock TrainingEnvironment for testing strategies.

    This function returns a MagicMock configured to behave like a TrainingEnvironment,
    satisfying the requirements of BaseTrainingStrategy.__init__.

    Args:
        config: Training configuration object (MagicMock or TrainingSettings).
        device: Device to use (cpu or cuda).
        generator: Generator model (optional, defaults to MagicMock).
        discriminator: Discriminator model (optional).
        opt_g: Generator optimizer (optional, defaults to MagicMock).
        opt_d: Discriminator optimizer (optional).
        model_type: Model type string for environment.

    Returns:
        MagicMock: A mock object mimicking TrainingEnvironment.
    """
    if isinstance(device, str):
        device = torch.device(device)

    env = MagicMock(spec=TrainingEnvironment)

    # Core attributes required by BaseTrainingStrategy
    env.config = config
    env.device = device

    # Models dictionary
    env.models = {}
    if generator:
        env.models["generator"] = generator
    else:
        # Default mock generator
        mock_gen = MagicMock(spec=nn.Module)
        mock_gen.training = True  # simulate training mode
        env.models["generator"] = mock_gen

    if discriminator:
        env.models["discriminator"] = discriminator
    elif model_type == "gan":
        # Create mock discriminator if GAN and none provided
        mock_disc = MagicMock(spec=nn.Module)
        mock_disc.training = True
        env.models["discriminator"] = mock_disc

    # Optimizers dictionary — canonical key is "opt_g" (see
    # TODO/backlog_standardize_optimizer_keys.md). The legacy "main"
    # alias was removed in 2026-05.
    env.optimizers = {}
    if opt_g:
        env.optimizers["opt_g"] = opt_g
    else:
        env.optimizers["opt_g"] = MagicMock(spec=Optimizer)

    if opt_d:
        env.optimizers["opt_d"] = opt_d

    # Property mocks (because BaseTrainingStrategy accesses properties)
    # MagicMock handles property access differently than instance attributes
    # We assign them as attributes because property access on a Mock returns a child Mock
    # but we want specific return values.

    # However, BaseTrainingStrategy accesses env.generator (property) which
    # executes code. But since env is a Mock, access to 'generator' returns a Mock.
    # We can configure the return value of the property descriptor if we really wanted to,
    # or just set the attribute directly if the code uses it like a property.
    # BaseTrainingStrategy uses `self.env.generator` which calls the property.
    # On a Mock, `env.generator` is just an attribute access.

    # Also set direct attributes for easier test setup if property mocking fails
    # (MagicMock doesn't always play nice with properties on the instance)
    env.generator = env.models.get("generator")
    env.discriminator = env.models.get("discriminator")
    env.opt_g = env.optimizers.get("opt_g")
    env.opt_d = env.optimizers.get("opt_d")
    env.model_type = model_type

    # Losses
    env.losses = {}
    env.loss_function = {}  # Alias

    return env
