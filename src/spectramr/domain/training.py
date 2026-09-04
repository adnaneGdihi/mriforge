from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from spectramr.domain.entities.data.types import MetricsDict, MRIBatchDict, OptimizerOrTuple


class TrainingStrategy(ABC):
    """
    Abstract Base Class for Training Strategies (GAN, Diffusion, Super-Resolution).
    Encapsulates the specific training logic for a single step/epoch.
    """

    @abstractmethod
    def train_step(
        self,
        batch: MRIBatchDict,
        model: nn.Module,
        optimizer: OptimizerOrTuple,
        device: torch.device,
        epoch: int,
        step: int,
        scaler: None = None,
    ) -> MetricsDict:
        """
        Execute a single training step.

        Args:
            batch: The input batch (dict, tuple, or tensor)
            model: The model to train (could be a wrapper containing G and D)
            optimizer: The optimizer(s) - could be a single optimizer or specific container
            device: Training device
            epoch: Current epoch
            step: Current global step

        Returns:
            logs: Dictionary of scalar metrics/losses to log
        """
        pass

    @abstractmethod
    def validation_step(
        self, batch: MRIBatchDict, model: nn.Module, device: torch.device
    ) -> MetricsDict:
        """
        Execute a single validation step.

        Note:
            Concrete implementations should wrap their body in
            ``@torch.no_grad()`` (or ``with torch.no_grad():``). Decorating this
            ABSTRACT method has no effect — the generated/overridden method is
            what actually runs, so the no-grad context must live on the override.

        Args:
            batch: Validation batch
            model: Model to evaluate
            device: Evaluation device

        Returns:
            logs: Dictionary of scalar metrics
        """
        pass
