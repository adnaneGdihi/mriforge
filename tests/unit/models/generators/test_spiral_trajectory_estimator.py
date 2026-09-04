"""SpiralTrajectoryEstimator — the load-bearing model for exp_vf_35 (G1).

Predicts the spiral trajectory deviation Δk̂(t) from a GIRF/delay parametrisation
θ: differentiate the nominal trajectory → apply the (learnable, identity-init)
GIRF → re-integrate → Δk̂ = k_act − k_nom. Identity GIRF ⇒ Δk̂ = 0 (the θ=θ0 /
Δk=0 consistency limit). Trained SUPERVISED against Δk_true (the GIRF path is
pure-torch differentiable; the NUFFT is not, so no through-NUFFT objective). The
estimate is the SPIRAL parametrisation [B, N, 2], not a Cartesian per-line [B,2,H].
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.spiral_trajectory_estimator import (
    SpiralTrajectoryEstimator,
)
from spectramr.models.registry import get_model_capabilities


def test_registered_with_spiral_parametrization():
    caps = get_model_capabilities("spiral_trajectory_estimator")
    assert caps is not None
    assert caps.trajectory_parametrization == "spiral"


def test_output_is_spiral_dk_same_shape_as_nominal():
    m = SpiralTrajectoryEstimator(in_channels=2, out_channels=2, girf_param_dim=21)
    k_nom = torch.randn(4, 512, 2)  # [n_acq, n_samp, 2] spiral coords
    dk = m(k_nom)
    assert dk.shape == (4, 512, 2)  # spiral Δk(t), NOT Cartesian [B,2,H]
    est = m.last_trajectory_estimate
    assert est is not None and est.shape == (4, 512, 2)


def test_identity_girf_yields_zero_deviation():
    # θ=θ0 (delta GIRF, zero k0) → Δk̂ ≈ 0 (consistency check 2 / the Δk=0 baseline).
    m = SpiralTrajectoryEstimator(in_channels=2, out_channels=2, girf_param_dim=21)
    k_nom = torch.cumsum(torch.randn(2, 256, 2) * 0.01, dim=1)  # smooth spiral
    with torch.no_grad():
        dk = m(k_nom)
    assert float(dk.abs().max()) < 1e-4


def test_freeze_estimator_is_the_nominal_trajectory_baseline():
    # The Δk=0 / no-estimate baseline (pitfall #17): a frozen identity GIRF has no
    # trainable params and always emits Δk̂=0 (the nominal-trajectory control the
    # trainable arm must beat). The single differing knob is freeze_estimator.
    m = SpiralTrajectoryEstimator(girf_param_dim=21, freeze_estimator=True)
    assert m.get_parameter_count() == 0
    k_nom = torch.cumsum(torch.randn(2, 128, 2) * 0.01, dim=1)
    assert float(m(k_nom).abs().max()) < 1e-4


def test_supervised_dk_loss_trains_the_girf_params():
    # A supervised Δk loss flows gradients to the GIRF kernel (the working path).
    m = SpiralTrajectoryEstimator(in_channels=2, out_channels=2, girf_param_dim=21)
    k_nom = torch.cumsum(torch.randn(2, 128, 2) * 0.01, dim=1)
    dk_true = torch.randn(2, 128, 2) * 0.05
    loss = ((m(k_nom) - dk_true) ** 2).mean()
    loss.backward()
    girf_grads = [p.grad for n, p in m.named_parameters() if "girf" in n]
    assert any(g is not None and g.abs().sum() > 0 for g in girf_grads)
