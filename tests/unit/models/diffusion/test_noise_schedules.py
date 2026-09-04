"""Regression tests for noise_schedules silent-fallback fixes.

1. ``NoiseScheduleFactory.create_noise_schedule`` silently degraded an unknown
   ``schedule_type`` to cosine -> must now raise ValueError (pitfall #9).
2. ``NoiseScheduleManager.switch_schedule`` silently no-op'd on an unknown name
   -> must now raise KeyError (pitfall #15: silent no-op knob).
"""

import pytest

from spectramr.models.diffusion.noise_schedules import (
    NoiseScheduleFactory,
    NoiseScheduleManager,
)


def test_unknown_schedule_type_raises():
    with pytest.raises(ValueError, match="Unknown schedule_type"):
        NoiseScheduleFactory.create_noise_schedule(schedule_type="bogus", num_timesteps=8)


def test_known_schedule_type_constructs():
    sched = NoiseScheduleFactory.create_noise_schedule(schedule_type="cosine", num_timesteps=8)
    assert sched is not None


def test_switch_schedule_unknown_name_raises_keyerror():
    a = NoiseScheduleFactory.create_noise_schedule(schedule_type="cosine", num_timesteps=8)
    b = NoiseScheduleFactory.create_noise_schedule(schedule_type="linear", num_timesteps=8)
    mgr = NoiseScheduleManager({"cosine": a, "linear": b})

    before_name = mgr.current_schedule_name
    before_sched = mgr.current_schedule

    with pytest.raises(KeyError):
        mgr.switch_schedule("does_not_exist")

    # State must be unchanged after the failed switch.
    assert mgr.current_schedule_name == before_name
    assert mgr.current_schedule is before_sched


def test_switch_schedule_known_name_switches():
    a = NoiseScheduleFactory.create_noise_schedule(schedule_type="cosine", num_timesteps=8)
    b = NoiseScheduleFactory.create_noise_schedule(schedule_type="linear", num_timesteps=8)
    mgr = NoiseScheduleManager({"cosine": a, "linear": b})

    mgr.switch_schedule("linear")
    assert mgr.current_schedule_name == "linear"
    assert mgr.current_schedule is b
