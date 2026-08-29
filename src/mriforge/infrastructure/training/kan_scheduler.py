import logging
from typing import Any

import torch
import torch.nn as nn


class KANGridScheduler:
    """KANGridScheduler class."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        logger: logging.Logger = None,
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            optimizer (torch.optim.Optimizer): Description.
            config (dict[str, Any]): Description.
            logger (logging.Logger): Description.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Parse milestones
        callbacks_cfg = config.get("training", {}).get("callbacks", [])
        self.scheduler_cfg = None
        for cb in callbacks_cfg:
            if cb.get("type") == "KANGridScheduler":
                self.scheduler_cfg = cb
                break

        if self.scheduler_cfg:
            self.milestones = self.scheduler_cfg.get("milestones", [])
            self.grid_sizes = self.scheduler_cfg.get("grid_sizes", [])
            assert len(self.milestones) == len(self.grid_sizes), (
                "Milestones and grid_sizes must match"
            )
            self.milestone_map = dict(zip(self.milestones, self.grid_sizes, strict=False))
        else:
            self.milestone_map = {}

    def on_epoch_end(self, epoch: int) -> None:
        """on_epoch_end.

        Args:
            epoch (int): Description.
        """
        if epoch not in self.milestone_map:
            return

        new_grid_size = self.milestone_map[epoch]
        self.logger.info(f"[KANGridScheduler] Epoch {epoch}: Updating grid size to {new_grid_size}")

        # 1. Update Model Layers
        count = 0

        # Import internally to avoid circular imports if any
        from mriforge.models.layers.kan.fastkanconv import FastKANConvLayer

        for name, module in self.model.named_modules():
            if isinstance(module, FastKANConvLayer):
                module.update_grid_size(new_grid_size)
                count += 1

        self.logger.info(f"[KANGridScheduler] Updated {count} KAN layers.")

        # 2. Re-initialize Optimizer
        # Note: This discards momentum history. For spectral continuation strictly,
        # we might want to preserve it, but parameter shapes changed, so momentum tensors are invalid.
        # Re-creation is the standard approach for "Growing" architectures.

        self._reinit_optimizer()

    def _reinit_optimizer(self) -> None:
        # Extract params from config
        """_reinit_optimizer."""
        opt_config = self.config.get("optimization", {})
        optimizer_type = opt_config.get("optimizer_type", "adam")
        lr = opt_config.get("learning_rate", 1e-4)
        weight_decay = opt_config.get("weight_decay", 1e-5)

        # Create new optimizer
        new_optimizer = None
        params = self.model.parameters()

        from mriforge.infrastructure.training.builders import OptimizationBuilder

        new_optimizer = OptimizationBuilder.create_single_optimizer(
            params,
            learning_rate=lr,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
        )

        # Replace the optimizer in the trainer/engine
        # This requires the scheduler to have access to the object holding the optimizer reference
        # OR we modify the optimizer object in-place (difficult)
        # Here we assume the caller checks `scheduler.optimizer` or we update existing ref if possible

        # BETTER: The Engine should ask the scheduler "Do you have a new optimizer?"
        self.optimizer = new_optimizer
        self.logger.info("[KANGridScheduler] Optimizer re-initialized.")
