import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.training.interfaces import IOptimizerStepper

logger = logging.getLogger(__name__)


@dataclass
class TrainableEntity:
    """A model coupled with its optimization policy."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    stepper: IOptimizerStepper
    scheduler: Any | None = None
    ema: Any | None = None


@dataclass
class TrainingEnvironment:
    """The complete 'Ready-to-Run' context for a training loop."""

    generator: TrainableEntity
    discriminator: TrainableEntity | None
    device: torch.device
    config: Any  # TrainingConfigSchema
    loss_system: Any = None
    metrics_computer: Any = None
    train_loader: Any = None
    val_loader: Any = None

    @property
    def generator_model(self) -> torch.nn.Module:
        """Helper to access the underlying generator module."""
        return self.generator.model

    @property
    def opt_g(self) -> Any:
        """Helper to access the generator optimizer."""
        return self.generator.optimizer

    @property
    def discriminator_model(self) -> torch.nn.Module | None:
        """Helper to access the underlying discriminator module."""
        return self.discriminator.model if self.discriminator else None

    @property
    def opt_d(self) -> Any:
        """Helper to access the discriminator optimizer."""
        return self.discriminator.optimizer if self.discriminator else None

    @property
    def model_type(self) -> str:
        """Helper to get model type from config."""
        return self.config.model.model_type

    @property
    def model_info(self) -> dict[str, Any]:
        """Helper to get model info from config."""
        return self.config.model.model_dump()

    @property
    def loss_function(self) -> Any:
        """Helper to get the main loss function.

        Returns None if loss_system is not initialized.
        """
        if self.loss_system is None:
            return None
        if isinstance(self.loss_system, dict):
            return self.loss_system.get("loss_function")
        return self.loss_system

    @property
    def criterion_l1(self) -> nn.Module | None:
        """Helper to get L1 criterion from loss system."""
        try:
            loss_fn = self.loss_function
            if loss_fn and hasattr(loss_fn, "l1_loss"):
                return loss_fn.l1_loss
        except (AttributeError, TypeError) as _exc:
            logger.debug("Suppressed exception: %s", _exc)
        return None

    @property
    def criterion_mse(self) -> nn.Module | None:
        """Helper to get MSE criterion from loss system."""
        try:
            loss_fn = self.loss_function
            if loss_fn and hasattr(loss_fn, "mse_loss"):
                return loss_fn.mse_loss
        except (AttributeError, TypeError) as _exc:
            logger.debug("Suppressed exception: %s", _exc)
        return None

    @property
    def criterion_perceptual(self) -> nn.Module | None:
        """Helper to get perceptual criterion from loss system."""
        if isinstance(self.loss_system, dict):
            return self.loss_system.get("perceptual_loss_module")
        return None
