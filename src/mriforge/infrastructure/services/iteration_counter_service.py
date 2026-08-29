"""Iteration Counter Service: SSOT for step tracking across all services.

This service maintains the canonical iteration state for training:
- Global step count (across entire training)
- Epoch counter
- Steps within current epoch
- Methods to check if interval-based operations should run

All interval-based services (checkpoint, logging, validation, metrics) query
this service to determine if their operations should execute.

Architecture: Single Source of Truth (SSOT)
- Initialized once at training session start
- Incremented once per training step
- Queried by all services to check interval conditions
- Respects max_iterations and max_steps_per_epoch hard limits
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mriforge.config.schemas.training.base import BaseTrainingConfigSchema


class IterationCounterService:
    """Centralized iteration counter managing all step-based operations.

    This is the SSOT (Single Source of Truth) for step counting across all services.
    All periodic operations (checkpoint, logging, validation, metrics) query this
    service to determine if their intervals have been reached.

    Attributes:
        config: Training configuration with max_iterations and max_steps_per_epoch
        current_step: Global step count across entire training (incremented each step)
        current_epoch: Current epoch number (incremented at epoch start)
        steps_in_current_epoch: Steps completed in current epoch (reset at epoch start)
        total_steps_this_epoch: Total steps expected in current epoch

    ### Interval Decision Flow
    ```mermaid
    graph TD
        A[Check Event] --> B{Step % Interval == 0?}
        B -- No --> C[Wait]
        B -- Yes --> D{Current Step < Max?}
        D -- No --> E[Halt Event]
        D -- Yes --> F[Run: Checkpoint/Log/Eval]
    ```
    """

    def __init__(self, config: "BaseTrainingConfigSchema"):
        """Initialize iteration counter from config.

        Args:
            config: Training configuration (must have max_iterations, max_steps_per_epoch)
        """
        self.config = config
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.steps_in_current_epoch: int = 0
        self.total_steps_this_epoch: int = 0

    def increment(self) -> None:
        """Increment step counters.

        Called once per training step. Updates both global step and epoch-relative step.
        This is the primary update method - called in the inner training loop.

        Raises:
            RuntimeError: If increment would exceed max_iterations hard limit
        """
        # Check hard limit before incrementing
        if self.config.max_iterations > 0 and self.current_step >= self.config.max_iterations:
            raise RuntimeError(
                f"Cannot increment beyond max_iterations={self.config.max_iterations}. "
                f"Use should_stop() to check before calling increment()."
            )

        self.current_step += 1
        self.steps_in_current_epoch += 1

    def start_epoch(self, total_steps_in_epoch: int = 0) -> None:
        """Start new epoch.

        Called at the beginning of each epoch. Increments epoch counter and resets
        per-epoch step counter.

        Args:
            total_steps_in_epoch: Expected number of steps in this epoch (for tracking)
        """
        self.current_epoch += 1
        self.steps_in_current_epoch = 0
        self.total_steps_this_epoch = total_steps_in_epoch

    def end_epoch(self) -> None:
        """End current epoch.

        Called at the end of each epoch. Resets per-epoch counter for next epoch.
        (This is mainly for clarity - start_epoch() also resets the counter.)
        """
        self.steps_in_current_epoch = 0

    # ─────────────────────────────────────────────────────────────────────────────
    # Interval Checking Methods
    # Each returns True if the corresponding operation should run now
    # ─────────────────────────────────────────────────────────────────────────────

    def should_checkpoint(self, checkpoint_interval: int) -> bool:
        """Check if checkpoint should be saved now.

        Args:
            checkpoint_interval: Interval from config.checkpoint.save_interval

        Returns:
            True if (current_step % interval == 0) and within max_iterations limit

        Example:
            >>> counter = IterationCounterService(config)
            >>> counter.current_step = 1000
            >>> counter.should_checkpoint(checkpoint_interval=1000)
            True

            >>> counter.current_step = 1500
            >>> counter.should_checkpoint(checkpoint_interval=1000)
            False
        """
        # Check interval condition
        interval_met = self.current_step % checkpoint_interval == 0

        # Check max_iterations hard limit
        within_limit = (
            self.config.max_iterations < 0 or self.current_step < self.config.max_iterations
        )

        return interval_met and within_limit

    def should_log(self, log_interval: int) -> bool:
        """Check if metrics should be logged now.

        Args:
            log_interval: Interval from config.logging.log_interval

        Returns:
            True if (current_step % interval == 0) and within max_iterations limit
        """
        interval_met = self.current_step % log_interval == 0
        within_limit = (
            self.config.max_iterations < 0 or self.current_step < self.config.max_iterations
        )
        return interval_met and within_limit

    def should_validate(
        self,
        eval_interval: int,
        frequency_epochs: int = 1,
        use_epoch_based: bool = False,
    ) -> bool:
        """Check if validation should run now.

        Supports DUAL MODE:
        - Step-based: (current_step % eval_interval == 0)
        - Epoch-based: (current_epoch % frequency_epochs == 0)

        Args:
            eval_interval: Interval from config.validation.eval_interval (steps)
            frequency_epochs: Interval from config.validation.frequency_epochs (epochs)
            use_epoch_based: If True, use epoch-based check; if False, use step-based

        Returns:
            True if interval condition met and within max_iterations limit

        Example (Step-based):
            >>> counter.current_step = 1000
            >>> counter.should_validate(eval_interval=1000)
            True

        Example (Epoch-based):
            >>> counter.current_epoch = 5
            >>> counter.should_validate(eval_interval=1000, frequency_epochs=5,
            ...                         use_epoch_based=True)
            True  # (5 % 5 == 0)
        """
        # Check interval condition
        if use_epoch_based:
            interval_met = self.current_epoch % frequency_epochs == 0
        else:
            interval_met = self.current_step % eval_interval == 0

        # Check max_iterations hard limit
        within_limit = (
            self.config.max_iterations < 0 or self.current_step < self.config.max_iterations
        )

        return interval_met and within_limit

    def should_compute_metrics(self, metric_interval: int) -> bool:
        """Check if metrics should be computed now.

        Args:
            metric_interval: Interval from config.metrics.metric_interval

        Returns:
            True if (current_step % interval == 0) and within max_iterations limit
        """
        interval_met = self.current_step % metric_interval == 0
        within_limit = (
            self.config.max_iterations < 0 or self.current_step < self.config.max_iterations
        )
        return interval_met and within_limit

    def should_visualize(self, visualization_interval: int) -> bool:
        """Check if visualizations should be saved now.

        Args:
            visualization_interval: Interval from config.validation.visualization_interval

        Returns:
            True if (current_step % interval == 0) and within max_iterations limit
        """
        interval_met = self.current_step % visualization_interval == 0
        within_limit = (
            self.config.max_iterations < 0 or self.current_step < self.config.max_iterations
        )
        return interval_met and within_limit

    # ─────────────────────────────────────────────────────────────────────────────
    # Hard Stop Conditions
    # Return True when training should terminate
    # ─────────────────────────────────────────────────────────────────────────────

    def should_stop_global(self) -> bool:
        """Check if training should stop (global limit reached).

        This is the hard stop for entire training. When True, no more steps
        should be executed and training should terminate.

        Returns:
            True if max_iterations is set and current_step >= max_iterations

        Example:
            >>> counter = IterationCounterService(config)
            >>> config.max_iterations = 1000
            >>> counter.current_step = 1000
            >>> counter.should_stop_global()
            True

            >>> counter.current_step = 999
            >>> counter.should_stop_global()
            False
        """
        if self.config.max_iterations < 0:
            # max_iterations = -1 means unlimited
            return False

        return self.current_step >= self.config.max_iterations

    def should_stop_epoch(self) -> bool:
        """Check if current epoch should stop (per-epoch limit reached).

        This is the hard stop for current epoch only. When True, training should
        break out of the inner loop and start the next epoch.

        Returns:
            True if max_steps_per_epoch is set and steps in current epoch >= limit
        """
        if self.config.max_steps_per_epoch < 0:
            # max_steps_per_epoch = -1 means unlimited
            return False

        return self.steps_in_current_epoch >= self.config.max_steps_per_epoch

    # ─────────────────────────────────────────────────────────────────────────────
    # State Accessors (for logging and debugging)
    # ─────────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Get current counter state (for logging, checkpointing, etc.).

        [PRODUCTION FIX] Now includes RNG state for deterministic restart.

        Returns:
            Dictionary with current step, epoch, per-epoch state, and RNG state
        """
        import random

        import numpy as np
        import torch

        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "steps_in_current_epoch": self.steps_in_current_epoch,
            "total_steps_this_epoch": self.total_steps_this_epoch,
            "max_iterations": self.config.max_iterations,
            "max_steps_per_epoch": self.config.max_steps_per_epoch,
            # [PRODUCTION FIX] RNG state for deterministic restart
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            },
        }

    def restore_state(self, state: dict) -> None:
        """Restore counter state from checkpoint.

        [PRODUCTION FIX] Now restores RNG state for deterministic dataloader.

        Args:
            state: Dictionary with counter state (from get_state())
        """
        import random

        import numpy as np
        import torch

        self.current_step = state.get("current_step", 0)
        self.current_epoch = state.get("current_epoch", 0)
        self.steps_in_current_epoch = state.get("steps_in_current_epoch", 0)
        self.total_steps_this_epoch = state.get("total_steps_this_epoch", 0)

        # [PRODUCTION FIX] Restore RNG state for deterministic dataloader restart
        rng_state = state.get("rng_state")
        if rng_state:
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and rng_state["cuda"] is not None:
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng_state["cuda"])

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"IterationCounterService("
            f"step={self.current_step}, "
            f"epoch={self.current_epoch}, "
            f"steps_in_epoch={self.steps_in_current_epoch}, "
            f"max_iterations={self.config.max_iterations}, "
            f"max_steps_per_epoch={self.config.max_steps_per_epoch})"
        )
