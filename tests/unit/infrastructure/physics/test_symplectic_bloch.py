"""Tests for the symplectic Bloch integrator."""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.symplectic_bloch import (
    energy_drift,
    hat_so3,
    munthe_kaas_step,
    rodrigues_exp,
    strang_split_step,
    vee_so3,
)


def test_hat_vee_are_inverses() -> None:
    """vee(hat(v)) = v for any 3-vector."""
    v = torch.tensor([1.0, 2.0, 3.0])
    out = vee_so3(hat_so3(v))
    assert torch.allclose(out, v)


def test_hat_so3_produces_skew_symmetric() -> None:
    """hat(v) is skew-symmetric: A + A^T = 0."""
    v = torch.randn(5, 3)
    a = hat_so3(v)
    assert torch.allclose(a + a.transpose(-1, -2), torch.zeros_like(a))


def test_rodrigues_exp_is_rotation_matrix() -> None:
    """exp([v]_×) is orthogonal with det = +1."""
    torch.manual_seed(0)
    axis_angle = torch.randn(4, 3)
    r = rodrigues_exp(axis_angle)
    eye = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)
    dets = torch.linalg.det(r)
    assert torch.allclose(dets, torch.ones(4), atol=1e-5)


def test_munthe_kaas_preserves_magnetisation_norm() -> None:
    """Single-step precession is exactly norm-preserving."""
    m = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 1.0])  # field along z
    m_new = munthe_kaas_step(m, b, dt=1e-6, gamma=1.0)
    assert m_new.norm().item() == pytest.approx(1.0, abs=1e-6)


def test_munthe_kaas_zero_field_is_identity() -> None:
    """B = 0 → no rotation, magnetisation unchanged."""
    m = torch.tensor([0.3, -0.4, 0.5])
    b = torch.zeros(3)
    m_new = munthe_kaas_step(m, b, dt=0.1, gamma=1.0)
    assert torch.allclose(m_new, m, atol=1e-6)


def test_no_relaxation_limit_preserves_norm_over_long_train() -> None:
    """With T1 = T2 = ∞ (relaxation off) and no field, the integrator is exact."""
    m = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 0.01])
    trajectory = [m]
    for _ in range(1000):
        m = strang_split_step(
            m, b, dt=1e-4, t1=1e12, t2=1e12, gamma=1.0
        )
        trajectory.append(m)
    drift = energy_drift(torch.stack(trajectory), m0=1.0).item()
    assert drift < 1e-3


def test_strang_split_relaxes_xy_under_finite_t2() -> None:
    """With B=0 and T1=∞, T2 finite, |M_xy| decays exponentially."""
    m = torch.tensor([1.0, 0.0, 0.0])
    b = torch.zeros(3)
    decayed = strang_split_step(m, b, dt=0.5, t1=1e12, t2=1.0, gamma=1.0)
    expected_xy = math.exp(-0.5)
    assert decayed[0].item() == pytest.approx(expected_xy, abs=1e-4)


def test_munthe_kaas_precesses_in_larmor_direction() -> None:
    """dM/dt = γ M×B: a quarter turn of x̂ about B=+ẑ yields −ŷ (Larmor sense).

    Regression for the 2026-07 sign fix — the integrator previously used
    exp(+γΔt[B]_×) = exp of dM/dt = γ B×M, the negation of the stated law,
    rotating x̂→+ŷ instead of x̂→−ŷ.
    """
    m = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 1.0])
    gamma = 3.0
    dt = (math.pi / 2) / (gamma * 1.0)  # γ·dt·|B| = π/2 (quarter turn)
    out = munthe_kaas_step(m, b, dt, gamma=gamma)
    assert torch.allclose(out, torch.tensor([0.0, -1.0, 0.0]), atol=1e-5)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        hat_so3(torch.zeros(4))
    with pytest.raises(ValueError):
        vee_so3(torch.zeros(3, 4))
    with pytest.raises(ValueError):
        rodrigues_exp(torch.zeros(4))
    with pytest.raises(ValueError):
        munthe_kaas_step(torch.zeros(3), torch.zeros(2, 3), dt=1e-6)
    with pytest.raises(ValueError):
        munthe_kaas_step(torch.zeros(2), torch.zeros(2), dt=1e-6)
