import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class LossScheduler:
    """Manages dynamic loss weights (Curriculum Learning).

    Allows scheduling loss weights based on global step.
    Supports:
    - Linear Warmup (ramp from initial to final over N steps)
    - Sigmoid Rampup (smooth transition)
    - Constant (base value * scale)

    The schedule math (:meth:`compute_schedule`) is the reusable curriculum
    primitive behind the ``ramp`` action of
    :class:`~mriforge.infrastructure.training.loss_schedule_controller.LossScheduleController`.

    Usage (per-name schedules indexed by global step)::

        scheduler = LossScheduler({"lambda_perceptual": {...}})
        weights = scheduler.update_weights({"lambda_perceptual": 0.0}, global_step)

    Or the stateless math directly::

        w = LossScheduler.compute_schedule(
            {"type": "linear_warmup", "start_step": 5000,
             "warmup_steps": 1000, "initial_value": 0.0, "final_value": 0.1},
            base_value=0.1, step=current_step,
        )
    """

    def __init__(self, schedules: dict[str, dict[str, Any]] | None = None):
        """
        Args:
            schedules: Dictionary mapping loss names to schedule configs.
                       Example:
                       {
                           "lambda_perceptual": {
                               "type": "linear_warmup",
                               "start_step": 1000,
                               "warmup_steps": 5000,
                               "initial_value": 0.0,
                               "final_value": 1.0
                           }
                       }
        """
        self.schedules = schedules or {}
        self._cache = {}

    def get_value(self, name: str, current_value: float, step: int) -> float:
        """Get scheduled value for a parameter.

        Args:
            name: Loss parameter name (e.g. "lambda_perceptual")
            current_value: Base value from config (used as final_value if not specified)
            step: Current global training step

        Returns:
            Scheduled value
        """
        if name not in self.schedules:
            return current_value

        schedule = self.schedules[name]
        return self.compute_schedule(schedule, current_value, step)

    @staticmethod
    def compute_schedule(config: dict[str, Any], base_value: float, step: int) -> float:
        """Resolve a single scheduled value at ``step`` (stateless, reusable).

        Args:
            config: Schedule spec with ``type`` in
                {``constant``, ``linear_warmup``, ``sigmoid_rampup``}, plus
                ``start_step`` and (per type) ``warmup_steps``/``ramp_steps``,
                ``initial_value``, ``final_value``, ``scale``.
            base_value: Fallback used as ``final_value`` when not specified
                (and the multiplicand for ``constant``).
            step: Current global training step.

        Returns:
            The scheduled scalar at ``step``. Before ``start_step`` returns
            ``initial_value``; an unknown type falls through to ``base_value``.
        """
        schedule_type = config.get("type", "constant")

        # Common params
        start_step = config.get("start_step", 0)

        if step < start_step:
            return config.get("initial_value", 0.0)

        local_step = step - start_step

        if schedule_type == "constant":
            return base_value * config.get("scale", 1.0)

        elif schedule_type == "linear_warmup":
            warmup_steps = config.get("warmup_steps", 1000)
            init_val = config.get("initial_value", 0.0)
            final_val = config.get("final_value", base_value)

            if local_step >= warmup_steps:
                return final_val

            alpha = local_step / warmup_steps
            return init_val + alpha * (final_val - init_val)

        elif schedule_type == "sigmoid_rampup":
            ramp_steps = config.get("ramp_steps", 1000)
            init_val = config.get("initial_value", 0.0)
            final_val = config.get("final_value", base_value)

            if local_step >= ramp_steps:
                return final_val

            # Sigmoid in range [-6, 6] approx
            # Map [0, ramp_steps] -> [-6, 6]
            x = (local_step / ramp_steps) * 12.0 - 6.0
            sigmoid = 1 / (1 + math.exp(-x))

            return init_val + sigmoid * (final_val - init_val)

        return base_value

    def update_weights(self, weights: dict[str, float], step: int) -> dict[str, float]:
        """Update a dictionary of weights in-place (or return new dict)."""
        new_weights = {}
        for k, v in weights.items():
            new_weights[k] = self.get_value(k, v, step)
        return new_weights
