"""k_space_trajectory_rmse — RMS error of an estimated spiral Δk(t) vs measured.

The trajectory analogue of b0_field_rmse for exp_vf_35. Grades a model's
``Δk̂`` against the measured deviation ``Δk_true = k_measured − k_nominal`` in
consistent units (cycles/FOV). The shared guard's shape check catches the
diff_trajectory_opt facade — a Cartesian per-readout-line ``Δk [B,2,H]`` graded
against a spiral ``Δk(t) [n_samp,2]`` is incomparable and returns NaN (skip),
never a silent grade (pitfall #16).
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.k_space_trajectory_rmse import KSpaceTrajectoryRMSE
from mriforge.core.metrics.registry import MetricsRegistry


def test_registered_and_lower_is_better():
    m = MetricsRegistry.get("k_space_trajectory_rmse")
    assert m.name == "k_space_trajectory_rmse"
    assert m.higher_is_better is False


def test_perfect_match_is_zero():
    m = KSpaceTrajectoryRMSE()
    traj = torch.randn(512, 2)
    assert m(traj, traj.clone()) == 0.0


def test_constant_offset_rmse():
    m = KSpaceTrajectoryRMSE()
    est = torch.full((256, 2), 0.3)
    ref = torch.full((256, 2), 0.1)
    assert abs(m(est, ref) - 0.2) < 1e-5  # constant 0.2 cycles/FOV offset


def test_cartesian_vs_spiral_shape_mismatch_is_skipped():
    # The diff_trajectory_opt facade: Cartesian per-line [B,2,H] vs spiral [N,2].
    m = KSpaceTrajectoryRMSE()
    cartesian = torch.zeros(1, 2, 16)
    spiral = torch.zeros(512, 2)
    assert math.isnan(m(cartesian, spiral))


def test_missing_estimate_is_skipped():
    m = KSpaceTrajectoryRMSE()
    assert math.isnan(m(None, torch.zeros(512, 2)))
