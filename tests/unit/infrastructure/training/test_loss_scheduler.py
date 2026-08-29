"""Unit tests for LossScheduler.

Targets ``mriforge.infrastructure.training.loss_scheduler.LossScheduler``.

Properties verified:
- Unregistered names return the current value unchanged.
- ``constant`` schedule applies a scale factor.
- ``linear_warmup`` ramps from initial to final over exactly warmup_steps.
- ``sigmoid_rampup`` lies strictly between initial and final at mid-ramp.
- ``update_weights`` processes an entire weights dict.
- Pre-start steps return ``initial_value`` (boundary behaviour).
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_import_and_construct() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler()
    assert sched is not None
    # No schedules → any name returns its current value unchanged.
    assert sched.get_value("lambda_perceptual", 0.5, step=100) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Constant schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scale,base,expected",
    [
        (1.0, 2.0, 2.0),
        (0.5, 4.0, 2.0),
        (0.0, 10.0, 0.0),
    ],
)
def test_constant_schedule(scale: float, base: float, expected: float) -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler({"lam": {"type": "constant", "scale": scale}})
    assert sched.get_value("lam", base, step=999) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Linear warmup
# ---------------------------------------------------------------------------


def test_linear_warmup_at_zero_step_returns_initial() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "linear_warmup",
                "warmup_steps": 1000,
                "initial_value": 0.0,
                "final_value": 1.0,
            }
        }
    )
    assert sched.get_value("lam", 1.0, step=0) == pytest.approx(0.0)


def test_linear_warmup_at_full_steps_returns_final() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "linear_warmup",
                "warmup_steps": 500,
                "initial_value": 0.0,
                "final_value": 2.0,
            }
        }
    )
    assert sched.get_value("lam", 1.0, step=500) == pytest.approx(2.0)


def test_linear_warmup_midpoint() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "linear_warmup",
                "warmup_steps": 1000,
                "initial_value": 0.0,
                "final_value": 1.0,
            }
        }
    )
    # At step 500 (halfway), value should be 0.5
    val = sched.get_value("lam", 1.0, step=500)
    assert val == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Sigmoid rampup
# ---------------------------------------------------------------------------


def test_sigmoid_rampup_between_initial_and_final() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "sigmoid_rampup",
                "ramp_steps": 1000,
                "initial_value": 0.0,
                "final_value": 1.0,
            }
        }
    )
    mid_val = sched.get_value("lam", 1.0, step=500)
    assert 0.0 < mid_val < 1.0


def test_sigmoid_rampup_returns_final_after_ramp() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "sigmoid_rampup",
                "ramp_steps": 200,
                "initial_value": 0.0,
                "final_value": 3.0,
            }
        }
    )
    assert sched.get_value("lam", 1.0, step=200) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Pre-start boundary
# ---------------------------------------------------------------------------


def test_prestep_returns_initial_value() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "lam": {
                "type": "linear_warmup",
                "start_step": 1000,
                "warmup_steps": 500,
                "initial_value": 0.1,
                "final_value": 1.0,
            }
        }
    )
    # Before start_step, must return initial_value from the config.
    val = sched.get_value("lam", 1.0, step=999)
    assert val == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# update_weights
# ---------------------------------------------------------------------------


def test_update_weights_processes_all_keys() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler(
        {
            "a": {"type": "constant", "scale": 2.0},
        }
    )
    weights = {"a": 1.0, "b": 5.0}
    updated = sched.update_weights(weights, step=0)
    assert updated["a"] == pytest.approx(2.0)
    # "b" has no schedule → unchanged
    assert updated["b"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Edge: unknown schedule type falls back to base
# ---------------------------------------------------------------------------


def test_edge_unknown_schedule_type_returns_base() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler({"lam": {"type": "totally_unknown"}})
    assert sched.get_value("lam", 7.0, step=100) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Edge: empty schedules dict
# ---------------------------------------------------------------------------


def test_edge_empty_schedules_passthrough() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    sched = LossScheduler({})
    assert sched.get_value("anything", 3.14, step=9999) == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# compute_schedule — the public, stateless staticmethod (reused by the
# LossScheduleController ``ramp`` action).
# ---------------------------------------------------------------------------


def test_compute_schedule_staticmethod_callable_without_instance() -> None:
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    # Callable on the class directly (no instance) — the reuse contract.
    val = LossScheduler.compute_schedule(
        {"type": "linear_warmup", "start_step": 100, "warmup_steps": 400,
         "initial_value": 0.0, "final_value": 1.0},
        base_value=1.0, step=300,
    )
    assert val == pytest.approx(0.5, abs=1e-6)  # halfway through the ramp


def test_compute_schedule_matches_get_value() -> None:
    """The instance path delegates to the staticmethod, so they agree."""
    from mriforge.infrastructure.training.loss_scheduler import LossScheduler

    cfg = {"type": "sigmoid_rampup", "ramp_steps": 200,
           "initial_value": 0.0, "final_value": 3.0}
    sched = LossScheduler({"lam": cfg})
    assert sched.get_value("lam", 3.0, step=120) == pytest.approx(
        LossScheduler.compute_schedule(cfg, base_value=3.0, step=120)
    )
