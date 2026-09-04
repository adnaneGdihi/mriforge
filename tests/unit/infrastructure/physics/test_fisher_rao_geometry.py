"""Tests for Fisher-Rao information-geometry primitives."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.fisher_rao_geometry import (
    fisher_information_from_jacobian,
    fisher_norm,
    fisher_rao_velocity_loss,
    monte_carlo_fisher_estimator,
)


def test_fisher_matrix_is_positive_semidefinite() -> None:
    """g = J^T J / σ² is PSD by construction."""
    torch.manual_seed(0)
    j = torch.randn(10, 3)
    g = fisher_information_from_jacobian(j, sigma=0.5)
    # All eigenvalues ≥ 0.
    eigs = torch.linalg.eigvalsh(g)
    assert (eigs >= -1e-5).all()


def test_fisher_scales_with_sigma_inverse_squared() -> None:
    """g scales as 1/σ²."""
    j = torch.randn(8, 3)
    g1 = fisher_information_from_jacobian(j, sigma=1.0)
    g2 = fisher_information_from_jacobian(j, sigma=2.0)
    assert torch.allclose(g2, g1 / 4.0)


def test_fisher_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        fisher_information_from_jacobian(torch.zeros(5), sigma=1.0)
    with pytest.raises(ValueError):
        fisher_information_from_jacobian(torch.zeros(5, 3), sigma=0.0)


def test_fisher_norm_matches_quadratic_form() -> None:
    """||v||_g = sqrt(v^T g v) — direct quadratic-form check."""
    g = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    v = torch.tensor([1.0, 1.0])
    expected = math.sqrt(2.0 + 1.0 + 2.0 * 0.5)  # v^T g v = 4
    out = fisher_norm(v, g)
    assert out.item() == pytest.approx(expected, abs=1e-5)


def test_fisher_norm_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        fisher_norm(torch.zeros(3), torch.zeros(2, 2))


def test_monte_carlo_estimator_converges_with_M() -> None:
    """Standard error decreases as 1/sqrt(M)."""
    torch.manual_seed(0)
    M = 64
    j_samples = torch.randn(M, 6, 3)
    _, std_err = monte_carlo_fisher_estimator(j_samples, sigma=0.5)
    # Larger M → smaller standard error.
    larger_M = 256
    j_samples_big = torch.randn(larger_M, 6, 3)
    _, std_err_big = monte_carlo_fisher_estimator(j_samples_big, sigma=0.5)
    # Ratio of std errors should be roughly sqrt(M/M_big) = 0.5.
    assert std_err_big.mean().item() < std_err.mean().item()


def test_monte_carlo_estimator_invalid_shape() -> None:
    with pytest.raises(ValueError):
        monte_carlo_fisher_estimator(torch.zeros(4, 3), sigma=1.0)


def test_velocity_loss_zero_when_velocities_match() -> None:
    """L = 0 when predicted = target, regardless of metric."""
    v = torch.randn(8, 3)
    g = torch.eye(3).expand(8, -1, -1)
    loss = fisher_rao_velocity_loss(v, v, g)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_velocity_loss_positive_when_mismatch() -> None:
    v_pred = torch.zeros(8, 3)
    v_target = torch.ones(8, 3)
    g = torch.eye(3).expand(8, -1, -1)
    loss = fisher_rao_velocity_loss(v_pred, v_target, g)
    # ||1||² = 3 (with identity metric).
    assert loss.item() == pytest.approx(3.0, abs=1e-5)
