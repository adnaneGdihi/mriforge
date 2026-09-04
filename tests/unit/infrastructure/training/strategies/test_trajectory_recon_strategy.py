"""TrajectoryReconstructionStrategy (vf_35 G4).

Supervises the spiral Δk̂ against the measured deviation Δk_true = measured −
nominal (the only differentiable path), and grades it with the guarded
k_space_trajectory_rmse via _score_trajectory. The self-supervised sharpness
knob is read + validated: it RAISES while unimplemented (no silent no-op,
pitfall #15).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.trajectory_recon_strategy import (
    TrajectoryReconstructionStrategy,
)
from spectramr.models.generators.spiral_trajectory_estimator import (
    SpiralTrajectoryEstimator,
)


def _cfg(self_supervised_sharpness: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            trajectory_recon=SimpleNamespace(
                read_measured_trajectory=True,
                trajectory_units="cycles_per_fov",
                girf_param_dim=21,
                self_supervised_sharpness=self_supervised_sharpness,
                sharpness_weight=0.0,
            )
        )
    )


def _strategy(self_supervised_sharpness: bool = False) -> TrajectoryReconstructionStrategy:
    st = TrajectoryReconstructionStrategy.__new__(TrajectoryReconstructionStrategy)
    st.config = _cfg(self_supervised_sharpness)
    st.device = torch.device("cpu")
    st._to_device = lambda t: t  # type: ignore[method-assign]
    st._setup_strategy_specific_components()
    st.env = SimpleNamespace(
        generator=SpiralTrajectoryEstimator(girf_param_dim=21)
    )
    return st


def _traj_pair(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    k_nom = torch.cumsum(torch.randn(2, 128, 2, generator=g) * 0.01, dim=1)
    k_meas = k_nom + torch.randn(2, 128, 2, generator=g) * 0.02
    return k_nom, k_meas


def test_compute_losses_supervised_traj_mse_is_differentiable():
    st = _strategy()
    k_nom, k_meas = _traj_pair(1)
    batch = {"trajectory_nominal": k_nom, "trajectory_measured": k_meas}
    img = torch.zeros(2, 1, 16, 16)
    out = st._compute_losses_impl(img, img, 0, batch=batch)
    assert "g_total_loss" in out
    assert out["g_total_loss"].requires_grad
    out["g_total_loss"].backward()  # gradients reach the GIRF params


def test_validation_emits_guarded_traj_rmse():
    st = _strategy()
    k_nom, k_meas = _traj_pair(2)
    img = torch.zeros(2, 1, 16, 16)
    val = st.validation_step(
        img, img, trajectory_measured=k_meas, trajectory_nominal=k_nom
    )
    assert val["val_traj_reference_real"] == 1.0
    assert "val_traj_rmse" in val
    assert val["val_traj_rmse"] == val["val_traj_rmse"]  # not NaN — shapes match


def test_score_trajectory_noop_without_reference():
    st = _strategy()
    assert st._score_trajectory(None, None) == {}


def test_self_supervised_sharpness_unimplemented_raises():
    with pytest.raises(ValueError, match="self_supervised_sharpness"):
        _strategy(self_supervised_sharpness=True)


def test_validation_step_declares_trajectory_kwargs_for_gating():
    # The trainer's gated pass (train.py) threads a batch key to validation_step
    # ONLY if the method declares the parameter — pin the contract.
    import inspect

    params = inspect.signature(
        TrajectoryReconstructionStrategy.validation_step
    ).parameters
    assert "trajectory_measured" in params
    assert "trajectory_nominal" in params
