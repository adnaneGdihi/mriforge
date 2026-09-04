"""Early stopping service for training monitoring.

This service monitors validation metrics and stops training when no improvement
is seen for a configured number of validation checks. The service is iteration-based,
not epoch-based, for finer control during long training runs.
"""

import logging
import math
from collections.abc import Callable

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.enums import MetricMode  # re-exported (see __all__)
from spectramr.infrastructure.training.plateau_monitor import PlateauMonitor

logger = logging.getLogger(__name__)


class EarlyStoppingService:
    """Early stopping monitor to halt training when metric plateaus.

    Features:
    - Iteration-based monitoring (not epoch-based)
    - Configurable patience and min_delta
    - Support for min/max metric modes
    - Callbacks on improvement
    - Checkpoint integration

    ### Stopping Decision Flow
    ```mermaid
    graph TD
        A[New Metric Value] --> B{Better than Best?}
        B -- Yes --> C[Reset Wait Count]
        C --> D[Update Best Value]
        D --> E[Trigger Callback]
        B -- No --> F[Increment Wait Count]
        F --> G{Patience Reached?}
        G -- Yes --> H[Set Stop Signal]
        G -- No --> I[Continue Training]
        E --> I
    ```
    """

    def __init__(
        self,
        config: EarlyStoppingConfigSchema,
        on_improvement: Callable[[int, float], None] | None = None,
    ):
        """Initialize Early Stopping Service.

        Args:
            config: Early stopping configuration Pydantic object (REQUIRED).
            on_improvement: Optional callback when metric improves (iteration, metric_value)
        """
        self.on_improvement = on_improvement

        # SSOT: direct access to Pydantic schema fields
        self.enabled = config.enabled
        self.monitor = config.metric
        self.patience = config.patience
        self.patience_min_iterations = config.patience_min_iterations
        self.mode = config.mode
        self.min_delta = config.min_delta
        self.check_interval = config.check_interval

        self._assert_mode_matches_metric_direction()

        # Plateau detection (strict comparator + best/wait tracking) is delegated
        # to the shared :class:`PlateauMonitor` so the comparator that guards
        # identity-collapse (a flat metric must never read as improvement) has a
        # single source of truth across early-stopping and loss scheduling.
        # ``cooldown=0``: early stopping fires once and latches ``stop_signal``,
        # it does not repeat. The iteration-based ``patience_min_iterations`` and
        # the latched ``stop_signal`` semantics stay on this service.
        self._monitor = PlateauMonitor(
            mode=self.mode, patience=self.patience, min_delta=self.min_delta
        )
        self.stop_signal = False

        logger.info(
            f"Early stopping initialized: monitor={self.monitor}, "
            f"patience={self.patience}, "
            f"min_iterations={self.patience_min_iterations}, "
            f"mode={self.mode.value}, "
            f"min_delta={self.min_delta}, check_interval={self.check_interval}"
        )

    @property
    def best_value(self) -> float:
        """Best monitored value seen so far (delegated to the plateau monitor)."""
        return self._monitor.best_value

    @property
    def best_iteration(self) -> int:
        """Iteration at which the best value was recorded."""
        return self._monitor.best_iteration

    @property
    def wait_count(self) -> int:
        """Consecutive non-improving checks since the last improvement."""
        return self._monitor.wait_count

    def _assert_mode_matches_metric_direction(self) -> None:
        """Refuse a ``mode`` that contradicts the metric-direction SSOT (#712).

        Selection direction has two sources. ``early_stopping.mode`` is what the
        service and the best-checkpoint block at ``training_loop.py`` actually
        obey; ``metric_directions.metric_higher_is_better`` is what every other
        consumer obeys (the computer, checkpoint retention, the leaderboard
        ranker). Nothing reconciled them, so a `mode` that disagreed silently
        retained the WORST checkpoint -- #208's outcome surviving at a new site
        after its resolver was fixed.

        Measured before writing this: all 922 arms declaring
        ``early_stopping.metric`` set ``mode`` explicitly, and exactly ONE
        contradicts the SSOT. So this is a guard against a class, not a rescue --
        stated plainly because "415 val_psnr arms select the worst checkpoint" was
        the intuitive reading and it is wrong.

        The YAML value is treated as an ASSERTION to be checked, never as a second
        source of truth: an undeclared direction is left alone (many monitor keys
        are strategy-emitted scalars), and only an outright contradiction raises.
        """
        if not self.enabled or not self.monitor:
            return

        from spectramr.core.metrics.metric_directions import resolve_direction

        declared_higher = str(self.mode).lower().endswith("max")
        ssot_higher = resolve_direction(self.monitor)
        if ssot_higher is None or ssot_higher == declared_higher:
            return

        raise ValueError(
            f"early_stopping.mode={self.mode!r} contradicts the metric-direction "
            f"SSOT for '{self.monitor}', which is declared "
            f"{'higher' if ssot_higher else 'lower'}-is-better.\n"
            "Selection would keep the WORST checkpoint for this metric. Either fix "
            "`mode`, or -- if the metric's declared direction is the wrong one -- "
            "fix it in core/metrics/metric_directions.py so every consumer agrees."
        )

    def should_check(self, iteration: int) -> bool:
        """Check if early stopping should be evaluated at this iteration.

        Args:
            iteration: Current training iteration

        Returns:
            True if early stopping should be checked at this iteration
        """
        return iteration > 0 and iteration % self.check_interval == 0

    def update(self, metric_value: float, iteration: int) -> None:
        """Update early stopping monitor with new metric value.

        Args:
            metric_value: Current metric value
            iteration: Current iteration (NOT epoch)
        """
        if not self.enabled:
            return

        # A non-finite metric (NaN/Inf) is not evidence of a plateau: ignore it
        # so it neither accrues patience nor (via the iteration-based
        # patience_min_iterations path, which reads best_iteration) trips a
        # spurious stop. The monitor itself also guards observe(); this guard
        # additionally protects the patience_min_iterations branch below.
        if not math.isfinite(metric_value):
            logger.debug(
                "Early stopping: ignoring non-finite metric %s at iteration %d",
                metric_value,
                iteration,
            )
            return

        # Strict-comparator improvement + wait tracking is delegated to the monitor.
        if self._monitor.observe(metric_value, iteration):
            logger.info(
                f"Early stopping: metric improved to {metric_value:.6f} at iteration {iteration}"
            )
            # Trigger callback
            if self.on_improvement:
                self.on_improvement(iteration, metric_value)
        else:
            logger.debug(f"Early stopping: no improvement ({self.wait_count}/{self.patience})")

        # Check if we should stop
        if self.patience_min_iterations is not None:
            # Iteration-based patience (User Request: "rely only on iterations")
            time_since_improvement = iteration - self.best_iteration
            if time_since_improvement >= self.patience_min_iterations:
                self.stop_signal = True
                logger.warning(
                    f"Early stopping triggered: no improvement for {time_since_improvement} iterations "
                    f"(limit: {self.patience_min_iterations}). "
                    f"Best value: {self.best_value:.6f} at iteration {self.best_iteration}"
                )
        elif self._monitor.is_plateaued():
            # Check-based patience (Legacy/Default)
            self.stop_signal = True
            logger.warning(
                f"Early stopping triggered: no improvement for {self.patience} checks. "
                f"Best value: {self.best_value:.6f} at iteration {self.best_iteration}"
            )

    def should_stop(self) -> bool:
        """Check if training should stop.

        Returns:
            True if training should stop, False otherwise
        """
        return self.stop_signal and self.enabled

    def reset(self) -> None:
        """Reset early stopping state."""
        self._monitor.reset()
        self.stop_signal = False
        logger.info("Early stopping state reset")

    def get_state(self) -> dict:
        """Get current state for logging.

        Returns:
            Dictionary with state information
        """
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode.value,
            "patience": self.patience,
            "patience_min_iterations": self.patience_min_iterations,
            "min_delta": self.min_delta,
            "check_interval": self.check_interval,
            "best_value": self.best_value,
            "best_iteration": self.best_iteration,
            "wait_count": self.wait_count,
            "should_stop": self.should_stop(),
        }


# MetricMode is re-exported for back-compat: callers/tests import it from here
# alongside EarlyStoppingService. Its canonical home is config.schemas.enums.
__all__ = ["EarlyStoppingService", "MetricMode"]
