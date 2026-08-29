"""Tier-1 audit checks for the trajectory-recovery arm (vf_35 Phase 3e).

Two static locks mirror the runtime guard:
* k_space_trajectory_rmse requires the model to declare
  trajectory_parametrization == "spiral" (the Cartesian-vs-spiral facade guard);
* read_measured_trajectory requires the dataset to emit the paired trajectories.
"""
from __future__ import annotations

from types import SimpleNamespace

from mriforge.infrastructure.validation.config_health_checker import ConfigHealthChecker
from tests.utils.config_block_stub import block_stub


def _cfg(metrics, model_type, *, read_measured=False, emit_paired=False):
    return SimpleNamespace(
        validation=block_stub("validation", metrics=metrics),
        model=SimpleNamespace(model_type=model_type),
        training=SimpleNamespace(
            trajectory_recon=SimpleNamespace(read_measured_trajectory=read_measured)
        ),
        data=SimpleNamespace(
            ismrmrd=SimpleNamespace(emit_paired_trajectory=emit_paired)
        ),
    )


def test_traj_metric_with_spiral_model_passes():
    chk = ConfigHealthChecker()
    r = chk.check_trajectory_metric_requires_matching_parametrization(
        _cfg(["k_space_trajectory_rmse"], "spiral_trajectory_estimator")
    )
    assert r.passed


def test_traj_metric_with_cartesian_model_errors():
    # diff_trajectory_opt is Cartesian per-line — grading it as spiral is the facade.
    chk = ConfigHealthChecker()
    r = chk.check_trajectory_metric_requires_matching_parametrization(
        _cfg(["k_space_trajectory_rmse"], "diff_trajectory_opt")
    )
    assert not r.passed
    assert r.severity == "error"


def test_traj_metric_not_requested_is_not_applicable():
    chk = ConfigHealthChecker()
    r = chk.check_trajectory_metric_requires_matching_parametrization(
        _cfg(["psnr"], "diff_trajectory_opt")
    )
    assert r.passed


def test_read_measured_requires_emit_paired():
    chk = ConfigHealthChecker()
    bad = chk.check_trajectory_recon_requires_measured_emit(
        _cfg(["k_space_trajectory_rmse"], "spiral_trajectory_estimator",
             read_measured=True, emit_paired=False)
    )
    assert not bad.passed and bad.severity == "error"
    good = chk.check_trajectory_recon_requires_measured_emit(
        _cfg(["k_space_trajectory_rmse"], "spiral_trajectory_estimator",
             read_measured=True, emit_paired=True)
    )
    assert good.passed
