from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class IOptimizerStepper(Protocol):
    """Encapsulates optimization policies like AMP, scaling, and clipping."""

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss for mixed precision."""
        ...

    def step(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor | None = None) -> bool:
        """Steps the optimizer and updates any internal state (like GradScaler)."""
        ...

    def perform_backward(self, loss: torch.Tensor) -> None:
        """Performs backpropagation, handling scaling if necessary."""
        ...

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Clears the gradients of the optimizer."""
        ...


@runtime_checkable
class ITrainableEntity(Protocol):
    """A model coupled with its optimization policy."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    stepper: IOptimizerStepper


@runtime_checkable
class ITrainingEnvironment(Protocol):
    """The complete 'Ready-to-Run' context for a training loop."""

    generator: ITrainableEntity
    discriminator: ITrainableEntity | None
    device: torch.device
    # we'll add more components as we implement them


class ITrainingState(ABC):
    """Interface for training state that holds model, config, and data."""

    device: torch.device
    model_type: str

    # Core components expected by strategies
    generator: nn.Module
    opt_g: torch.optim.Optimizer

    # Optional components
    discriminator: nn.Module | None
    opt_d: torch.optim.Optimizer | None

    # Configuration
    config: Any

    @property
    @abstractmethod
    def epoch(self) -> int:
        """epoch.

        Returns:
            int: Description.
        """
        pass

    @property
    @abstractmethod
    def global_step(self) -> int:
        """global_step.

        Returns:
            int: Description.
        """
        pass


class ITrainingStrategy(ABC):
    """Interface for training strategies."""

    @abstractmethod
    def train_step(self, *args, **kwargs) -> dict[str, Any]:
        """Execute a training step."""
        pass

    @property
    @abstractmethod
    def state(self) -> ITrainingState:
        """Access associated training state."""
        pass
