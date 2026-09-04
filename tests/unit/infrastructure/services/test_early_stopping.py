"""Tests for ``EarlyStoppingService``.

Targets ``spectramr.infrastructure.services.early_stopping``. The service
monitors validation metrics and signals when training should halt.
Iteration-based, supports min/max modes, optional improvement
callbacks, and either check-based or iteration-based patience.

Categories:

- Disabled service is a no-op
- ``mode=MIN`` monitors decreasing metrics; improvements reset wait
- ``mode=MAX`` monitors increasing metrics
- ``min_delta`` threshold prevents trivial improvements from resetting
- ``patience`` triggers ``should_stop`` after exhausted checks
- ``patience_min_iterations`` overrides check-based patience
- ``on_improvement`` callback fires on improvement
- ``reset`` returns to initial state
- ``should_check`` respects ``check_interval``
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
from spectramr.config.schemas.enums import MetricMode
from spectramr.infrastructure.services.early_stopping import EarlyStoppingService


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_persists_config_fields() -> None:
    """Service stores all config fields on the instance."""
    cfg = EarlyStoppingConfigSchema(
        enabled=True,
        patience=5,
        metric="val_loss",
        min_delta=0.01,
        check_interval=100,
        mode=MetricMode.MIN,
    )
    svc = EarlyStoppingService(config=cfg)
    assert svc.enabled is True
    assert svc.patience == 5
    assert svc.monitor == "val_loss"
    assert svc.min_delta == 0.01
    assert svc.check_interval == 100
    assert svc.mode == MetricMode.MIN


def test_constructor_initialises_min_mode_with_inf() -> None:
    """In MIN mode, initial best is +inf so any finite value improves it."""
    cfg = EarlyStoppingConfigSchema(enabled=True, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    assert svc.best_value == float("inf")


def test_constructor_initialises_max_mode_with_neg_inf() -> None:
    """In MAX mode, initial best is -inf."""
    cfg = EarlyStoppingConfigSchema(enabled=True, metric="val_psnr", mode=MetricMode.MAX)
    svc = EarlyStoppingService(cfg)
    assert svc.best_value == float("-inf")


# ---------------------------------------------------------------------------
# Disabled service
# ---------------------------------------------------------------------------


def test_disabled_service_update_is_noop() -> None:
    """When ``enabled=False``, ``update`` doesn't change state."""
    cfg = EarlyStoppingConfigSchema(enabled=False)
    svc = EarlyStoppingService(cfg)
    svc.update(metric_value=999.0, iteration=100)
    assert svc.best_value == float("inf")  # unchanged
    assert svc.wait_count == 0


def test_disabled_service_should_stop_always_false() -> None:
    """``should_stop`` is always False when disabled."""
    cfg = EarlyStoppingConfigSchema(enabled=False, patience=1)
    svc = EarlyStoppingService(cfg)
    # Force the stop_signal manually — should_stop still returns False.
    svc.stop_signal = True
    assert svc.should_stop() is False


# ---------------------------------------------------------------------------
# MIN mode behaviour
# ---------------------------------------------------------------------------


def test_min_mode_lower_metric_resets_wait() -> None:
    """In MIN mode, a lower metric → improvement → wait reset."""
    cfg = EarlyStoppingConfigSchema(enabled=True, patience=3, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)
    assert svc.best_value == 1.0
    svc.update(0.5, iteration=20)  # improvement
    assert svc.best_value == 0.5
    assert svc.wait_count == 0


def test_min_mode_higher_metric_increments_wait() -> None:
    """In MIN mode, a higher metric → no improvement → wait_count += 1."""
    cfg = EarlyStoppingConfigSchema(enabled=True, patience=5, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)  # initial
    svc.update(2.0, iteration=20)  # worse
    svc.update(3.0, iteration=30)  # worse
    assert svc.wait_count == 2


# ---------------------------------------------------------------------------
# MAX mode behaviour
# ---------------------------------------------------------------------------


def test_max_mode_higher_metric_resets_wait() -> None:
    """In MAX mode (e.g. PSNR), higher is better."""
    cfg = EarlyStoppingConfigSchema(enabled=True, metric="val_psnr", mode=MetricMode.MAX)
    svc = EarlyStoppingService(cfg)
    svc.update(20.0, iteration=10)
    svc.update(25.0, iteration=20)  # improvement
    assert svc.best_value == 25.0
    assert svc.wait_count == 0


# ---------------------------------------------------------------------------
# min_delta threshold
# ---------------------------------------------------------------------------


def test_min_delta_blocks_trivial_improvements() -> None:
    """In MIN mode with min_delta=0.5, a 0.1 drop doesn't count as improvement."""
    cfg = EarlyStoppingConfigSchema(enabled=True, mode=MetricMode.MIN, min_delta=0.5, patience=10)
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)
    svc.update(0.9, iteration=20)  # only 0.1 better, doesn't count
    assert svc.best_value == 1.0
    assert svc.wait_count == 1


# ---------------------------------------------------------------------------
# Patience triggering
# ---------------------------------------------------------------------------


def test_patience_triggers_stop_signal() -> None:
    """After ``patience`` consecutive non-improvements, ``should_stop()`` is True."""
    cfg = EarlyStoppingConfigSchema(enabled=True, patience=2, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)  # initial
    svc.update(2.0, iteration=20)  # wait=1
    svc.update(3.0, iteration=30)  # wait=2 → trigger
    assert svc.should_stop() is True


def test_patience_min_iterations_overrides_check_count() -> None:
    """``patience_min_iterations`` triggers stop on iteration count, not check count."""
    cfg = EarlyStoppingConfigSchema(
        enabled=True,
        mode=MetricMode.MIN,
        patience=999,  # high (won't trigger first)
        patience_min_iterations=50,
    )
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)  # best
    svc.update(2.0, iteration=70)  # 60 iterations of no improvement → trigger
    assert svc.should_stop() is True


# ---------------------------------------------------------------------------
# on_improvement callback
# ---------------------------------------------------------------------------


def test_callback_fires_on_improvement() -> None:
    """``on_improvement`` is called on each improvement."""
    received = []

    def callback(iteration: int, value: float) -> None:
        received.append((iteration, value))

    cfg = EarlyStoppingConfigSchema(enabled=True, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg, on_improvement=callback)
    svc.update(1.0, iteration=10)
    svc.update(2.0, iteration=20)  # no improvement, no callback
    svc.update(0.5, iteration=30)  # improvement
    assert received == [(10, 1.0), (30, 0.5)]


# ---------------------------------------------------------------------------
# should_check
# ---------------------------------------------------------------------------


def test_should_check_respects_interval() -> None:
    """``should_check`` returns True only every ``check_interval`` iterations."""
    cfg = EarlyStoppingConfigSchema(enabled=True, check_interval=100)
    svc = EarlyStoppingService(cfg)
    assert svc.should_check(0) is False  # iteration 0 excluded
    assert svc.should_check(50) is False
    assert svc.should_check(100) is True
    assert svc.should_check(200) is True
    assert svc.should_check(150) is False


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_returns_to_initial_state() -> None:
    """``reset`` clears wait/stop state and resets best to ±inf."""
    cfg = EarlyStoppingConfigSchema(enabled=True, mode=MetricMode.MIN, patience=1)
    svc = EarlyStoppingService(cfg)
    svc.update(1.0, iteration=10)
    svc.update(2.0, iteration=20)
    svc.update(3.0, iteration=30)
    assert svc.should_stop() is True

    svc.reset()
    assert svc.best_value == float("inf")
    assert svc.wait_count == 0
    assert svc.stop_signal is False
    assert svc.should_stop() is False


# ---------------------------------------------------------------------------
# get_state — for logging
# ---------------------------------------------------------------------------


def test_delegates_to_plateau_monitor() -> None:
    """After the refactor the service composes a shared PlateauMonitor; its
    best_value/wait_count/best_iteration are properties reading that monitor."""
    from spectramr.infrastructure.training.plateau_monitor import PlateauMonitor

    cfg = EarlyStoppingConfigSchema(enabled=True, patience=3, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    assert isinstance(svc._monitor, PlateauMonitor)
    svc.update(1.0, iteration=10)
    svc.update(2.0, iteration=20)
    # property reflects the monitor's live state
    assert svc.wait_count == svc._monitor.wait_count == 1
    assert svc.best_value == svc._monitor.best_value == 1.0
    assert svc.best_iteration == svc._monitor.best_iteration == 10


def test_get_state_returns_documented_keys() -> None:
    """``get_state`` returns a dict with all documented keys."""
    cfg = EarlyStoppingConfigSchema(enabled=True, mode=MetricMode.MIN)
    svc = EarlyStoppingService(cfg)
    state = svc.get_state()
    expected = {
        "enabled",
        "monitor",
        "mode",
        "patience",
        "patience_min_iterations",
        "min_delta",
        "check_interval",
        "best_value",
        "best_iteration",
        "wait_count",
        "should_stop",
    }
    assert expected == state.keys()


def test_nan_metric_is_ignored_and_never_stops() -> None:
    """A non-finite metric must not accrue patience nor trip the iteration-based
    patience_min_iterations stop (M7)."""
    from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
    from spectramr.config.schemas.enums import MetricMode
    from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

    cfg = EarlyStoppingConfigSchema(
        enabled=True, patience=2, metric="val_psnr", mode=MetricMode.MAX, patience_min_iterations=10
    )
    svc = EarlyStoppingService(cfg)
    for it in (1, 5, 20, 50, 100):
        svc.update(float("nan"), it)
    assert svc.wait_count == 0
    assert svc.should_stop() is False


# ---------------------------------------------------------------------------
# #712: `mode` and the metric-direction SSOT are now reconciled.
#
# Selection direction had two sources. `early_stopping.mode` is what this service
# and the best-checkpoint block in training_loop actually obey;
# `metric_directions.metric_higher_is_better` is what every OTHER consumer obeys.
# Nothing reconciled them, so a disagreeing `mode` silently retained the WORST
# checkpoint — #208's outcome surviving at a new site after its resolver was fixed.
#
# Measured before writing this: all 922 arms declaring `early_stopping.metric` set
# `mode` explicitly, and exactly ONE contradicts the SSOT. A guard against a class,
# not a rescue.
# ---------------------------------------------------------------------------


class TestModeIsReconciledWithTheDirectionSSOT:
    @staticmethod
    def _build(metric, mode, *, enabled=True):
        from spectramr.config.schemas.early_stopping import EarlyStoppingConfigSchema
        from spectramr.infrastructure.services.early_stopping import EarlyStoppingService

        return EarlyStoppingService(
            EarlyStoppingConfigSchema(enabled=enabled, metric=metric, mode=mode, patience=3)
        )

    @pytest.mark.parametrize(
        ("metric", "mode"),
        [
            ("val_psnr", "min"),  # higher-is-better minimized
            ("val_lpips", "max"),  # lower-is-better maximized
            ("val_combined", "max"),  # the one real corpus case
        ],
    )
    def test_a_contradiction_raises(self, metric, mode):
        # Matched loosely on purpose. The contradiction now has TWO guards and
        # the schema's runs first: ``EarlyStoppingConfigSchema`` refuses it at
        # config load (6dbd12c7b), so ``EarlyStoppingService``'s own
        # ``_assert_mode_matches_metric_direction`` -- whose wording this regex
        # used to pin -- is unreachable with a real schema object. Pinning the
        # earlier message asserts WHICH owner wins, which is exactly the
        # duplication still to be resolved; pinning the refusal asserts the
        # behaviour that must hold either way.
        with pytest.raises(ValueError, match="contradicts"):
            self._build(metric, mode)

    @pytest.mark.parametrize(
        ("metric", "mode"),
        [
            ("val_psnr", "max"),
            ("val_lpips", "min"),
            ("val_loss", "min"),
            ("val_field_abs_bias", "min"),
        ],
    )
    def test_agreement_passes(self, metric, mode):
        assert self._build(metric, mode) is not None

    @pytest.mark.parametrize("metric", ["some_strategy_scalar", "grad_norm"])
    def test_an_undeclared_direction_is_left_alone(self, metric):
        """Many monitor keys are strategy-emitted scalars with no declaration.

        The YAML value is treated as an ASSERTION to check, never as a second
        source of truth — so "no declaration" means "no opinion", not "raise".
        """
        assert self._build(metric, "min") is not None

    def test_disabled_early_stopping_is_not_policed(self):
        """A disabled block's `mode` decides nothing, so it cannot be wrong."""
        assert self._build("val_psnr", "min", enabled=False) is not None
