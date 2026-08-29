"""Unit tests for the timestep-acceleration schedule logic.

Captures the bug found 2026-05-05: when a user picks
``schedule_type=step`` + ``base_acceleration=2.0`` but does not declare
``acceleration_range``, the base-class auto-default
``[1.0, max_acceleration]`` silently overrode ``base_acceleration`` so
the schedule returned 1.0 (no undersampling) for the first half of
training instead of 2x. The fix tracks whether ``acceleration_range``
was explicitly provided.
"""

from __future__ import annotations

import math

import pytest

from mriforge.infrastructure.physics.sampling import UniformCartesianKSpaceAccelerator


def _sampler(**kw):
    defaults = dict(
        num_timesteps=1000,
        max_acceleration=32.0,
        base_acceleration=2.0,
        center_fraction=0.08,
        seed=42,
    )
    defaults.update(kw)
    return UniformCartesianKSpaceAccelerator(**defaults)


class TestLinearSchedule:
    def test_endpoints(self):
        s = _sampler(acceleration_schedule="linear")
        assert s.get_acceleration_factor(0) == pytest.approx(2.0, abs=0.01)
        assert s.get_acceleration_factor(999) == pytest.approx(32.0, abs=0.01)

    def test_monotonic(self):
        s = _sampler(acceleration_schedule="linear")
        prev = -math.inf
        for t in range(0, 1000, 50):
            cur = s.get_acceleration_factor(t)
            assert cur >= prev
            prev = cur


class TestPowerLawSchedule:
    def test_slow_start_fast_end(self):
        s = _sampler(acceleration_schedule="power_law", schedule_power=2.0)
        # At t=100/1000, ratio=0.1 → ratio² = 0.01 → accel ≈ 2.3 (close to base)
        assert s.get_acceleration_factor(100) < 2.5
        # At t=999, ratio=1 → accel=max
        assert s.get_acceleration_factor(999) == pytest.approx(32.0, abs=0.01)

    def test_endpoints(self):
        s = _sampler(acceleration_schedule="power_law", schedule_power=2.0)
        assert s.get_acceleration_factor(0) == pytest.approx(2.0, abs=0.01)
        assert s.get_acceleration_factor(999) == pytest.approx(32.0, abs=0.01)


class TestExponentialSchedule:
    def test_endpoints(self):
        s = _sampler(acceleration_schedule="exponential")
        assert s.get_acceleration_factor(0) == pytest.approx(2.0, abs=0.01)
        assert s.get_acceleration_factor(999) == pytest.approx(32.0, abs=0.01)


class TestStepBinarySchedule:
    """Critical bug: previously returned 1.0 not base for ratio<0.5."""

    def test_first_half_returns_base_not_one(self):
        s = _sampler(acceleration_schedule="step")
        # Without explicit acceleration_range, the binary fallback
        # should use base/max — the bug returned 1.0 here.
        for t in (0, 100, 250, 499):
            assert s.get_acceleration_factor(t) == pytest.approx(2.0, abs=0.01)

    def test_second_half_returns_max(self):
        s = _sampler(acceleration_schedule="step")
        for t in (500, 750, 999):
            assert s.get_acceleration_factor(t) == pytest.approx(32.0, abs=0.01)

    def test_explicit_range_overrides_binary(self):
        s = _sampler(
            acceleration_schedule="step",
            acceleration_range=[2.0, 4.0, 8.0, 16.0, 32.0],
        )
        # 5 buckets across 1000 steps → bucket boundaries every 200 steps
        assert s.get_acceleration_factor(0) == pytest.approx(2.0, abs=0.01)
        assert s.get_acceleration_factor(250) == pytest.approx(4.0, abs=0.01)
        assert s.get_acceleration_factor(450) == pytest.approx(8.0, abs=0.01)
        assert s.get_acceleration_factor(650) == pytest.approx(16.0, abs=0.01)
        assert s.get_acceleration_factor(999) == pytest.approx(32.0, abs=0.01)


class TestTargetSamplingFraction:
    """frac = 1/accel — verify the relationship across schedules."""

    @pytest.mark.parametrize("sched", ["linear", "power_law", "exponential", "step"])
    def test_frac_inverse_of_accel(self, sched):
        s = _sampler(acceleration_schedule=sched)
        for t in (0, 250, 500, 750, 999):
            accel = s.get_acceleration_factor(t)
            frac = s._target_sampling_fraction(t)
            assert frac == pytest.approx(1.0 / accel, rel=0.01)


class TestNumTimestepsHandling:
    def test_t_clamped_at_top(self):
        s = _sampler(acceleration_schedule="linear", num_timesteps=100)
        assert s.get_acceleration_factor(150) == s.get_acceleration_factor(99)

    def test_t_clamped_at_bottom(self):
        s = _sampler(acceleration_schedule="linear")
        assert s.get_acceleration_factor(-5) == s.get_acceleration_factor(0)
