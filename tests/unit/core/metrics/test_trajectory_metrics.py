"""Trajectory monitoring (κ_s trace, s*, excursions) and its cohort statistics.

Every expectation here is planted analytically: the κ and distance values are
constructed so the correct answer is a hand-computable number, never a
regression snapshot.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.core.metrics.trajectory_metrics import (
    TrajectoryMonitor,
    calibrate_admissible_radius,
    hallucination_rate,
    severity,
)
from mriforge.infrastructure.calibration.chd import dkw_slack


def _y_and_mask() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(1, 2, 16, 16, generator=generator)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::2] = 1.0
    return full * mask, mask


class TestKappaTrace:
    def test_data_consistent_steps_have_zero_kappa(self):
        y, mask = _y_and_mask()
        monitor = TrajectoryMonitor(y, mask)
        for step in (3, 2, 1, 0):
            # Prediction agrees with the measurement ON the observed support;
            # arbitrary content off-support must not register.
            pred = y + torch.randn_like(y) * (1.0 - mask)
            monitor(step, pred)
        assert monitor.kappa_per_step == [0.0, 0.0, 0.0, 0.0]
        assert monitor.first_violation_index() is None

    def test_kappa_is_scale_invariant(self):
        y, mask = _y_and_mask()
        pred = y * 1.3 + 0.1 * (1.0 - mask)
        small = TrajectoryMonitor(y, mask)
        small(0, pred)
        large = TrajectoryMonitor(10.0 * y, mask)
        large(0, 10.0 * pred)
        assert small.kappa_per_step[0] == pytest.approx(large.kappa_per_step[0], rel=1e-6)

    def test_planted_violation_at_step_k_yields_s_star_k(self):
        """κ = ‖2y·M‖/‖y·M‖ = 2 at the planted step, 0 elsewhere."""
        y, mask = _y_and_mask()
        monitor = TrajectoryMonitor(y, mask, kappa_threshold=0.5)
        for position, step in enumerate((6, 4, 2, 0)):
            pred = 3.0 * y if position == 2 else y
            monitor(step, pred)
        assert monitor.kappa_per_step[2] == pytest.approx(2.0, rel=1e-6)
        assert monitor.first_violation_index() == 2

    def test_summary_carries_the_trace(self):
        y, mask = _y_and_mask()
        monitor = TrajectoryMonitor(y, mask)
        monitor(1, 2.0 * y)
        monitor(0, y)
        summary = monitor.summary()
        assert summary["num_steps"] == 2
        assert summary["step_indices"] == [1, 0]
        assert summary["trajectory_kappa_final"] == 0.0
        assert summary["trajectory_kappa_max"] == pytest.approx(1.0, rel=1e-6)
        assert "distance_per_step" not in summary  # no target configured


class TestExcursion:
    def test_clean_trajectory_has_zero_excursion(self):
        y, mask = _y_and_mask()
        target = torch.ones(1, 2, 16, 16)
        monitor = TrajectoryMonitor(y, mask, admissible_radius=0.5, target=target)
        for step in (1, 0):
            monitor(step, target.clone())
        assert monitor.excursion() == 0.0
        summary = monitor.summary()
        assert summary["trajectory_excursion"] == 0.0
        assert summary["trajectory_max_violation_ratio"] == 0.0
        assert summary["first_violation_index"] is None

    def test_planted_excursion_is_exact(self):
        """pred = 1.5·target ⇒ d = 0.5; radius 0.2 ⇒ excursion 0.3, MVR 2.5."""
        y, mask = _y_and_mask()
        target = torch.ones(1, 2, 16, 16)
        monitor = TrajectoryMonitor(y, mask, admissible_radius=0.2, target=target)
        monitor(1, target.clone())
        monitor(0, 1.5 * target)
        assert monitor.excursion() == pytest.approx(0.3, rel=1e-5)
        summary = monitor.summary()
        assert summary["trajectory_max_violation_ratio"] == pytest.approx(2.5, rel=1e-5)
        assert monitor.first_violation_index() == 1

    def test_radius_without_target_raises(self):
        y, mask = _y_and_mask()
        with pytest.raises(ValueError, match="target"):
            TrajectoryMonitor(y, mask, admissible_radius=0.5)

    def test_nonpositive_radius_raises(self):
        y, mask = _y_and_mask()
        with pytest.raises(ValueError, match="admissible_radius"):
            TrajectoryMonitor(y, mask, admissible_radius=0.0, target=y)


class TestCohortStatistics:
    def test_hallucination_rate_counts_positive_excursions(self):
        assert hallucination_rate([0.0, 0.0, 0.1, 0.4]) == 0.5
        assert hallucination_rate([0.0, 0.0]) == 0.0

    def test_hallucination_rate_empty_raises(self):
        with pytest.raises(ValueError):
            hallucination_rate([])

    def test_severity_is_the_tail_mean(self):
        values = [float(i) for i in range(1, 11)]  # 1..10
        # alpha=0.2 over n=10 -> worst 2 -> mean(10, 9) = 9.5
        assert severity(values, alpha=0.2) == pytest.approx(9.5)
        # alpha=1.0 -> plain mean
        assert severity(values, alpha=1.0) == pytest.approx(5.5)

    def test_severity_validates_inputs(self):
        with pytest.raises(ValueError):
            severity([], alpha=0.1)
        with pytest.raises(ValueError, match="alpha"):
            severity([1.0], alpha=0.0)


class TestCalibrateAdmissibleRadius:
    def test_conformal_quantile_on_a_known_grid(self):
        """n=100 distances i/100: rank ⌈101·0.95⌉ = 96 ⇒ ϱ = 0.96."""
        distances = [i / 100 for i in range(1, 101)]
        rho, band = calibrate_admissible_radius(distances, alpha=0.05)
        assert rho == pytest.approx(0.96)
        assert band == pytest.approx(dkw_slack(100, 0.05))

    def test_small_sample_clips_to_the_maximum(self):
        """n=10, alpha=0.05: ⌈11·0.95⌉ = 11 > n ⇒ ϱ = max = 1.0."""
        distances = [i / 10 for i in range(1, 11)]
        rho, band = calibrate_admissible_radius(distances, alpha=0.05)
        assert rho == pytest.approx(1.0)
        assert band == pytest.approx(math.sqrt(math.log(2 / 0.05) / 20))

    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            calibrate_admissible_radius([], alpha=0.05)
        with pytest.raises(ValueError, match="alpha"):
            calibrate_admissible_radius([0.1], alpha=1.5)
