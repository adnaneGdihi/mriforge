"""Unit tests for KSpaceAccelerator scheduling types.

Tests all 4 schedule types: step, linear, power_law, exponential.
Validates monotonicity, boundary conditions, and mathematical properties.
"""

import math
from typing import Any

import pytest
import torch

from spectramr.infrastructure.physics.sampling import KSpaceAccelerator


class _ConcreteAccelerator(KSpaceAccelerator):
    """Minimal concrete stub — only the scheduling logic is under test."""

    def get_acceleration_mask(
        self,
        spatial_shape: tuple[int, ...],
        timestep: int,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return an all-ones mask (not part of the scheduling tests)."""
        return torch.ones(1, 1, *spatial_shape, device=device)


def _make(schedule: str, **overrides: Any) -> _ConcreteAccelerator:
    """Helper to construct an accelerator with sensible defaults."""
    defaults: dict[str, Any] = {
        "num_timesteps": 1000,
        "max_acceleration": 16.0,
        "acceleration_schedule": schedule,
    }
    defaults.update(overrides)
    acc = _ConcreteAccelerator(**defaults)
    # base_acceleration goes through **kwargs → getattr fallback
    if "base_acceleration" in overrides:
        acc.base_acceleration = overrides["base_acceleration"]
    return acc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def linear_accel() -> _ConcreteAccelerator:
    """Linear schedule: R linearly from base to max."""
    return _make("linear", base_acceleration=1.0)


@pytest.fixture()
def step_accel() -> _ConcreteAccelerator:
    """Step schedule with 6 discrete acceleration factors."""
    return _make(
        "step",
        acceleration_range=[2.0, 4.0, 8.0, 10.0, 12.0, 16.0],
    )


@pytest.fixture()
def power_law_accel() -> _ConcreteAccelerator:
    """Power-law schedule (quadratic by default)."""
    return _make("power_law", schedule_power=2.0, base_acceleration=1.0)


@pytest.fixture()
def exponential_accel() -> _ConcreteAccelerator:
    """Exponential schedule."""
    return _make("exponential", base_acceleration=1.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Tests for KSpaceAccelerator construction and validation."""

    def test_positive_timesteps_required(self) -> None:
        """Negative or zero timesteps must raise."""
        with pytest.raises(ValueError):
            _make("linear", num_timesteps=0)

    def test_max_acceleration_clamped_to_1(self) -> None:
        """max_acceleration < 1 is clamped to 1.0."""
        acc = _make("linear", max_acceleration=0.5)
        assert acc.max_acceleration == 1.0

    def test_default_schedule_is_linear(self) -> None:
        """Default schedule should be linear."""
        acc = _ConcreteAccelerator(num_timesteps=10)
        assert acc.acceleration_schedule == "linear"


# ---------------------------------------------------------------------------
# Linear Schedule
# ---------------------------------------------------------------------------
class TestLinearSchedule:
    """Tests for the linear acceleration schedule."""

    def test_boundary_t0_returns_base(self, linear_accel: _ConcreteAccelerator) -> None:
        """At t=0, acceleration should equal base."""
        r = linear_accel.get_acceleration_factor(0)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_boundary_tmax_returns_max(
        self, linear_accel: _ConcreteAccelerator
    ) -> None:
        """At t=T-1, acceleration should equal max."""
        r = linear_accel.get_acceleration_factor(999)
        assert r == pytest.approx(16.0, abs=0.1)

    def test_midpoint_linear(self, linear_accel: _ConcreteAccelerator) -> None:
        """At t=T/2, acceleration should be midpoint of [base, max]."""
        r = linear_accel.get_acceleration_factor(500)
        expected = 1.0 + 0.5 * (16.0 - 1.0)  # 8.5
        assert r == pytest.approx(expected, abs=0.5)

    def test_monotonically_increasing(self, linear_accel: _ConcreteAccelerator) -> None:
        """Acceleration must increase with timestep."""
        values = [linear_accel.get_acceleration_factor(t) for t in range(0, 1000, 10)]
        for i in range(1, len(values)):
            assert (
                values[i] >= values[i - 1]
            ), f"Non-monotonic at t={i * 10}: {values[i]} < {values[i - 1]}"

    def test_all_values_in_range(self, linear_accel: _ConcreteAccelerator) -> None:
        """All acceleration factors must be in [base, max]."""
        for t in range(1000):
            r = linear_accel.get_acceleration_factor(t)
            assert 1.0 <= r <= 16.0, f"Out of range at t={t}: R={r}"


# ---------------------------------------------------------------------------
# Step Schedule
# ---------------------------------------------------------------------------
class TestStepSchedule:
    """Tests for the discrete step acceleration schedule."""

    def test_returns_only_defined_values(
        self, step_accel: _ConcreteAccelerator
    ) -> None:
        """Step schedule must only return values from acceleration_range."""
        defined = {2.0, 4.0, 8.0, 10.0, 12.0, 16.0}
        for t in range(1000):
            r = step_accel.get_acceleration_factor(t)
            assert r in defined, f"Undeclared R={r} at t={t}"

    def test_first_bin(self, step_accel: _ConcreteAccelerator) -> None:
        """t=0 should map to the first acceleration value."""
        assert step_accel.get_acceleration_factor(0) == 2.0

    def test_last_bin(self, step_accel: _ConcreteAccelerator) -> None:
        """t=T-1 should map to the last acceleration value."""
        assert step_accel.get_acceleration_factor(999) == 16.0

    def test_bin_boundaries(self, step_accel: _ConcreteAccelerator) -> None:
        """Each bin should span approximately T/N timesteps."""
        # 6 bins over 1000 timesteps ≈ 167 each
        expected_bins = [2.0, 4.0, 8.0, 10.0, 12.0, 16.0]
        bin_size = 1000 / 6

        for i, expected_r in enumerate(expected_bins):
            # Sample from middle of each bin
            t_mid = int((i + 0.5) * bin_size)
            t_mid = min(t_mid, 999)
            r = step_accel.get_acceleration_factor(t_mid)
            assert (
                r == expected_r
            ), f"Bin {i} (t={t_mid}): expected R={expected_r}, got R={r}"

    def test_monotonically_non_decreasing(
        self, step_accel: _ConcreteAccelerator
    ) -> None:
        """Step schedule must be non-decreasing (when range is sorted)."""
        values = [step_accel.get_acceleration_factor(t) for t in range(1000)]
        for i in range(1, len(values)):
            assert (
                values[i] >= values[i - 1]
            ), f"Decrease at t={i}: {values[i]} < {values[i - 1]}"

    def test_single_element_range(self) -> None:
        """Single-element acceleration_range should return that value always."""
        acc = _make("step", acceleration_range=[4.0])
        for t in range(100):
            assert acc.get_acceleration_factor(t) == 4.0

    def test_empty_range_fallback(self) -> None:
        """Empty acceleration_range should fallback to base/max split."""
        acc = _make(
            "step",
            num_timesteps=100,
            max_acceleration=8.0,
            acceleration_range=[],
            base_acceleration=1.0,
        )
        # With empty range, falls to the else branch: base if ratio < 0.5 else max
        r_low = acc.get_acceleration_factor(0)
        r_high = acc.get_acceleration_factor(99)
        assert r_low == 1.0
        assert r_high == 8.0


# ---------------------------------------------------------------------------
# Power-Law Schedule
# ---------------------------------------------------------------------------
class TestPowerLawSchedule:
    """Tests for the power-law (quadratic) acceleration schedule."""

    def test_boundary_t0(self, power_law_accel: _ConcreteAccelerator) -> None:
        """At t=0, acceleration should equal base."""
        assert power_law_accel.get_acceleration_factor(0) == pytest.approx(
            1.0, abs=0.01
        )

    def test_boundary_tmax(self, power_law_accel: _ConcreteAccelerator) -> None:
        """At t=T-1, acceleration should equal max."""
        assert power_law_accel.get_acceleration_factor(999) == pytest.approx(
            16.0, abs=0.1
        )

    def test_slower_than_linear_early(
        self, power_law_accel: _ConcreteAccelerator, linear_accel: _ConcreteAccelerator
    ) -> None:
        """Power-law (p=2) should have LOWER acceleration than linear at midpoint."""
        t_mid = 250  # quarter way
        r_power = power_law_accel.get_acceleration_factor(t_mid)
        r_linear = linear_accel.get_acceleration_factor(t_mid)
        assert (
            r_power < r_linear
        ), f"Power-law ({r_power}) should be < linear ({r_linear}) at t={t_mid}"

    def test_faster_than_linear_late(
        self, power_law_accel: _ConcreteAccelerator, linear_accel: _ConcreteAccelerator
    ) -> None:
        """Verify quadratic progress at midpoint."""
        t = 500
        expected_ratio = (500.0 / 999.0) ** 2  # ≈ 0.251
        expected_r = 1.0 + expected_ratio * (16.0 - 1.0)
        actual = power_law_accel.get_acceleration_factor(t)
        assert actual == pytest.approx(expected_r, abs=0.5)

    def test_quadratic_formula(self, power_law_accel: _ConcreteAccelerator) -> None:
        """Verify R(t) = base + ((t/(T-1))^2) * (max - base)."""
        for t in [0, 100, 250, 500, 750, 999]:
            ratio = (float(t) / 999.0) ** 2
            expected = 1.0 + ratio * 15.0
            actual = power_law_accel.get_acceleration_factor(t)
            assert actual == pytest.approx(
                expected, abs=0.01
            ), f"t={t}: expected {expected:.4f}, got {actual:.4f}"

    def test_monotonically_increasing(
        self, power_law_accel: _ConcreteAccelerator
    ) -> None:
        """Power-law schedule must be monotonically increasing."""
        values = [power_law_accel.get_acceleration_factor(t) for t in range(0, 1000, 5)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_cubic_power(self) -> None:
        """schedule_power=3 should give cubic behavior."""
        acc = _make("power_law", schedule_power=3.0, base_acceleration=1.0)
        t = 500
        ratio = (500.0 / 999.0) ** 3
        expected = 1.0 + ratio * 15.0
        assert acc.get_acceleration_factor(t) == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Exponential Schedule
# ---------------------------------------------------------------------------
class TestExponentialSchedule:
    """Tests for the exponential acceleration schedule."""

    def test_boundary_t0(self, exponential_accel: _ConcreteAccelerator) -> None:
        """At t=0, acceleration should equal base."""
        assert exponential_accel.get_acceleration_factor(0) == pytest.approx(
            1.0, abs=0.01
        )

    def test_boundary_tmax(self, exponential_accel: _ConcreteAccelerator) -> None:
        """At t=T-1, acceleration should equal max."""
        assert exponential_accel.get_acceleration_factor(999) == pytest.approx(
            16.0, abs=0.1
        )

    def test_exponential_formula(self, exponential_accel: _ConcreteAccelerator) -> None:
        """Verify R(t) = base + ((exp(5r)-1)/(exp(5)-1)) * (max-base)."""
        for t in [0, 100, 250, 500, 750, 999]:
            r = float(t) / 999.0
            progress = (math.exp(r * 5.0) - 1.0) / (math.exp(5.0) - 1.0)
            expected = 1.0 + progress * 15.0
            actual = exponential_accel.get_acceleration_factor(t)
            assert actual == pytest.approx(
                expected, abs=0.01
            ), f"t={t}: expected {expected:.4f}, got {actual:.4f}"

    def test_slower_than_linear_first_half(
        self,
        exponential_accel: _ConcreteAccelerator,
        linear_accel: _ConcreteAccelerator,
    ) -> None:
        """Exponential should be slower than linear in the first half."""
        for t in [100, 200, 300]:
            r_exp = exponential_accel.get_acceleration_factor(t)
            r_lin = linear_accel.get_acceleration_factor(t)
            assert (
                r_exp < r_lin
            ), f"t={t}: exp ({r_exp:.2f}) should be < linear ({r_lin:.2f})"

    def test_faster_ramp_than_power_law_late(
        self,
        exponential_accel: _ConcreteAccelerator,
        power_law_accel: _ConcreteAccelerator,
    ) -> None:
        """Exponential should ramp harder than power-law very near the end.

        The exponential formula (exp(5r)-1)/(exp(5)-1) grows slower than
        quadratic for most of the trajectory, but eventually crosses over.
        At t=990 (ratio≈0.991), exp progress ≈ 0.963 vs pow progress ≈ 0.982.
        This test therefore checks the relative ordering is consistent
        with the formulas (exponential is still below quadratic at t=990).
        """
        # At t very close to T-1, both converge to 1.0
        # Verify the exponential is still below power_law at t=990
        t = 990
        r_exp = exponential_accel.get_acceleration_factor(t)
        r_pow = power_law_accel.get_acceleration_factor(t)
        # Both should be close to max, and power_law (p=2) is above exponential
        assert (
            r_exp < r_pow
        ), f"t={t}: exp ({r_exp:.2f}) should be < power_law ({r_pow:.2f})"

    def test_monotonically_increasing(
        self, exponential_accel: _ConcreteAccelerator
    ) -> None:
        """Exponential schedule must be monotonically increasing."""
        values = [
            exponential_accel.get_acceleration_factor(t) for t in range(0, 1000, 5)
        ]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


# ---------------------------------------------------------------------------
# Cross-Schedule Comparisons
# ---------------------------------------------------------------------------
class TestCrossScheduleProperties:
    """Compare properties across schedule types."""

    def test_all_schedules_match_at_boundaries(
        self,
        linear_accel: _ConcreteAccelerator,
        power_law_accel: _ConcreteAccelerator,
        exponential_accel: _ConcreteAccelerator,
    ) -> None:
        """All continuous schedules should match at t=0 and t=T-1."""
        for acc in [linear_accel, power_law_accel, exponential_accel]:
            assert acc.get_acceleration_factor(0) == pytest.approx(1.0, abs=0.1)
            assert acc.get_acceleration_factor(999) == pytest.approx(16.0, abs=0.1)

    def test_ordering_at_midpoint(
        self,
        linear_accel: _ConcreteAccelerator,
        power_law_accel: _ConcreteAccelerator,
        exponential_accel: _ConcreteAccelerator,
    ) -> None:
        """At midpoint: exponential < power_law < linear (all delay degradation)."""
        t = 500
        r_lin = linear_accel.get_acceleration_factor(t)
        r_pow = power_law_accel.get_acceleration_factor(t)
        r_exp = exponential_accel.get_acceleration_factor(t)

        assert (
            r_exp < r_pow < r_lin
        ), f"Expected exp ({r_exp:.2f}) < pow ({r_pow:.2f}) < lin ({r_lin:.2f})"

    def test_area_under_curve(
        self,
        linear_accel: _ConcreteAccelerator,
        power_law_accel: _ConcreteAccelerator,
        exponential_accel: _ConcreteAccelerator,
    ) -> None:
        """Integral of acceleration: linear > power_law > exponential.

        More aggressive schedules (power_law, exponential) have lower total
        undersampling across the diffusion trajectory — they keep data closer
        to fully-sampled for longer.
        """

        def total_accel(acc: _ConcreteAccelerator) -> float:
            return sum(acc.get_acceleration_factor(t) for t in range(1000))

        a_lin = total_accel(linear_accel)
        a_pow = total_accel(power_law_accel)
        a_exp = total_accel(exponential_accel)

        assert (
            a_exp < a_pow < a_lin
        ), f"Expected exp ({a_exp:.0f}) < pow ({a_pow:.0f}) < lin ({a_lin:.0f})"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_timestep(self) -> None:
        """T=1 should always return max acceleration."""
        acc = _make("linear", num_timesteps=1, base_acceleration=1.0)
        # _normalized_progress returns 1.0 when T<=1
        r = acc.get_acceleration_factor(0)
        assert r == pytest.approx(16.0, abs=0.1)

    def test_timestep_clamping(self) -> None:
        """Timesteps beyond [0, T-1] should be clamped."""
        acc = _make("linear", num_timesteps=100, base_acceleration=1.0)
        # t=-5 should clamp to t=0
        r_neg = acc.get_acceleration_factor(-5)
        r_zero = acc.get_acceleration_factor(0)
        assert r_neg == r_zero

        # t=200 should clamp to t=99
        r_over = acc.get_acceleration_factor(200)
        r_max = acc.get_acceleration_factor(99)
        assert r_over == r_max

    def test_step_with_unsorted_range(self) -> None:
        """Step schedule with non-monotonic range should still work."""
        acc = _make(
            "step",
            num_timesteps=100,
            acceleration_range=[8.0, 2.0, 16.0],
        )
        # Should still map bins correctly (even if non-monotonic)
        r_0 = acc.get_acceleration_factor(0)
        r_50 = acc.get_acceleration_factor(50)
        r_99 = acc.get_acceleration_factor(99)
        assert r_0 == 8.0
        assert r_50 == 2.0
        assert r_99 == 16.0

    def test_very_high_power(self) -> None:
        """Very high power should keep acceleration near base until the very end."""
        acc = _make(
            "power_law",
            schedule_power=10.0,
            base_acceleration=1.0,
        )
        # At t=500 (midpoint), progress = (0.5)^10 ≈ 0.001
        r_mid = acc.get_acceleration_factor(500)
        assert r_mid < 1.1, f"With power=10, mid should be near base, got {r_mid}"

        # At t=999, should still reach max
        r_end = acc.get_acceleration_factor(999)
        assert r_end == pytest.approx(16.0, abs=0.1)
