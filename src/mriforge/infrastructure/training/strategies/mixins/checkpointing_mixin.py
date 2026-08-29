"""Checkpointing Mixin

This module contains the logic for state save/load and checkpoint management.
"""

import logging
from typing import Any

from mriforge.domain.interfaces.service_interfaces import ICheckpointService
from mriforge.infrastructure.di.di_container import resolve_service

logger = logging.getLogger(__name__)


class CheckpointingMixin:
    """Mixin for state save/load and checkpoint management."""

    @property
    def checkpoint_service(self) -> ICheckpointService:
        """Lazy-loaded checkpoint service."""
        if not hasattr(self, "_checkpoint_service") or self._checkpoint_service is None:
            self._checkpoint_service = resolve_service(ICheckpointService)
        return self._checkpoint_service

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        loss: float,
        metrics: dict[str, float] | None = None,
        extra_optimizers: dict[str, Any] | None = None,
    ) -> None:
        """Save a checkpoint of the current training state.

        Args:
            epoch: Current epoch number.
            step: Current global step.
            loss: Current loss value.
            metrics: Optional dictionary of metrics.
            extra_optimizers: Optional dictionary of additional optimizers to save.
        """
        try:
            # Prepare extra optimizers if not provided, for example if discriminator exists
            if extra_optimizers is None and hasattr(self.env, "opt_d") and self.env.opt_d:
                extra_optimizers = {"discriminator_optimizer": self.env.opt_d}

            self.checkpoint_service.save_checkpoint(
                model=self.env.generator,
                optimizer=self.env.opt_g,
                epoch=epoch,
                step=step,
                loss=loss,
                metrics=metrics,
                extra_optimizers=extra_optimizers,
            )

            if hasattr(self, "logging_service"):
                self.logging_service.log_info(f"Checkpoint saved at epoch {epoch}, step {step}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if hasattr(self, "logging_service"):
                self.logging_service.log_warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load a checkpoint from the specified path.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The loaded checkpoint dictionary.
        """
        try:
            checkpoint = self.checkpoint_service.load_checkpoint(checkpoint_path)

            # Note: Applying the checkpoint to the model/optimizer is typically handled
            # by the ModelInitializer or Strategy explicitly.
            # Here we just return the loaded data.
            # If we wanted to load into self.state, we would do it here.

            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise e
