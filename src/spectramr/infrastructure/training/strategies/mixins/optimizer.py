"""Optimizer setup mixin for training strategies.

Owns the construction of the optimizer stepper. The runtime gradient
ops (`_zero_gradients`, `_clip_and_log_gradients`, `_backward_and_step`)
live on ``BaseTrainingStrategy`` itself — see CLAUDE.md "Change
discipline" / TODO/audit/05_strategies_core_mixins_builders.md F2 for
the history. Defining the dead siblings here advertised an MRO
contribution that never won dispatch, so they were removed.
"""

from typing import Any

from spectramr.infrastructure.training.optimizers.standard_optimizer_stepper import (
    StandardOptimizerStepper,
)


class OptimizerMixin:
    """Constructs the optimizer stepper used by ``BaseTrainingStrategy``."""

    amp_policy: Any
    config: Any

    def _build_optimizer_stepper(self, logging_service: Any) -> Any:
        """Build the optimizer stepper component.

        Args:
            logging_service: Logging service instance

        Returns:
            Configured optimizer stepper
        """
        # WS1-schemas-misc-03: thread the advertised ``optimization.
        # gradient_clip_method`` knob through to the stepper. Previously the
        # stepper always used its constructor default ("norm"), so the config
        # field was a silent no-op (pitfall #15) for any user selecting "value"
        # / "adaptive". ``self.config`` is provided by BaseTrainingStrategy.
        return StandardOptimizerStepper(
            max_grad_norm=self.amp_policy.max_grad_norm,
            enable_gradient_clipping=self.amp_policy.enable_gradient_clipping,
            gradient_clip_method=self.config.optimization.gradient.clip.method,
            logger=logging_service,
        )


__all__ = ["OptimizerMixin"]
