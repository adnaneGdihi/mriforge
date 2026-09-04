"""Tests for the Stein Variational federated aggregation service."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.services.stein_aggregation import (
    federated_svgd_step,
    product_of_experts_score,
    rbf_kernel,
    svgd_update_step,
)


def test_rbf_kernel_symmetric_with_ones_on_diagonal() -> None:
    """K(x, x) = 1; K is symmetric."""
    torch.manual_seed(0)
    particles = torch.randn(8, 4)
    kernel, _ = rbf_kernel(particles, bandwidth=1.0)
    diag = torch.diagonal(kernel)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
    assert torch.allclose(kernel, kernel.T, atol=1e-6)


def test_rbf_kernel_gradient_zero_on_diagonal() -> None:
    """∇_{x_i} K(x_i, x_i) = 0 (the gradient at the diagonal vanishes)."""
    particles = torch.randn(8, 4)
    _, grad_kernel = rbf_kernel(particles, bandwidth=1.0)
    diag_grad = torch.diagonal(grad_kernel, dim1=0, dim2=1).T  # (n, d)
    assert torch.allclose(diag_grad, torch.zeros_like(diag_grad), atol=1e-6)


def test_svgd_step_attracts_particles_toward_gaussian() -> None:
    """For a Gaussian target the SVGD step moves particles toward the mean."""
    torch.manual_seed(0)
    particles = torch.randn(32, 2) + torch.tensor([5.0, 0.0])  # offset
    target_mean = torch.zeros(2)

    def score(p: torch.Tensor) -> torch.Tensor:
        return -(p - target_mean)

    new_particles = svgd_update_step(particles, score, step_size=0.5)
    # Mean should shift toward the origin.
    initial_mean_dist = (particles.mean(dim=0) - target_mean).norm()
    new_mean_dist = (new_particles.mean(dim=0) - target_mean).norm()
    assert new_mean_dist < initial_mean_dist


def test_product_of_experts_score_averages_correctly() -> None:
    scores = [torch.full((4, 3), 2.0), torch.full((4, 3), 4.0)]
    out = product_of_experts_score(scores)
    assert torch.allclose(out, torch.full((4, 3), 3.0))


def test_federated_svgd_step_runs_without_dp() -> None:
    torch.manual_seed(0)
    particles = torch.randn(8, 4)
    scores = [-particles, -particles]
    new = federated_svgd_step(particles, scores, step_size=0.05, dp_noise_std=0.0)
    assert new.shape == particles.shape
    # Without DP and identical scores, the result is deterministic.
    new2 = federated_svgd_step(particles, scores, step_size=0.05, dp_noise_std=0.0)
    assert torch.allclose(new, new2)


def test_federated_svgd_dp_noise_adds_randomness() -> None:
    torch.manual_seed(0)
    particles = torch.randn(8, 4)
    scores = [-particles]
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(1)
    new1 = federated_svgd_step(particles, scores, dp_noise_std=0.1, generator=g1)
    new2 = federated_svgd_step(particles, scores, dp_noise_std=0.1, generator=g2)
    assert not torch.allclose(new1, new2)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        rbf_kernel(torch.zeros(8))  # 1-D particles
    with pytest.raises(ValueError):
        rbf_kernel(torch.zeros(1, 4))  # only 1 particle
    with pytest.raises(ValueError):
        svgd_update_step(torch.zeros(8, 2), lambda p: p, step_size=-0.1)
    with pytest.raises(ValueError):
        svgd_update_step(torch.zeros(8, 2), lambda p: torch.zeros(8, 3))
    with pytest.raises(ValueError):
        product_of_experts_score([])
    with pytest.raises(ValueError):
        product_of_experts_score([torch.zeros(4, 2), torch.zeros(4, 3)])
    with pytest.raises(ValueError):
        federated_svgd_step(torch.zeros(8, 2), [torch.zeros(8, 2)], dp_noise_std=-1.0)


# ---------------------------------------------------------------------------
# Repulsion sign (regression — kernel-gradient sign flip, 2026-06-12 audit)
# ---------------------------------------------------------------------------


def test_svgd_repulsive_term_pushes_particles_apart() -> None:
    """With a zero score the SVGD update is the pure repulsive force, which
    must INCREASE inter-particle separation (SVGD preserves multi-modality).

    Regression for the flipped RBF kernel-gradient sign: the buggy code
    computed ``(2/h)(x_i - x_j)k`` (the negation of ``∇_{x_i}k``), turning the
    repulsive force into an attractive one that collapses the particles.
    """
    particles = torch.tensor([[-0.5], [0.5]])

    def zero_score(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    updated = svgd_update_step(
        particles, zero_score, step_size=0.1, bandwidth=1.0
    )
    sep_before = (particles[1] - particles[0]).abs().item()
    sep_after = (updated[1] - updated[0]).abs().item()
    assert sep_after > sep_before, (
        f"repulsion must increase separation: {sep_before} -> {sep_after}"
    )


def test_rbf_grad_kernel_matches_analytic_sign() -> None:
    """``grad_kernel[i, j] == (2/h)(x_j - x_i) k`` (the docstring formula)."""
    particles = torch.tensor([[0.0], [1.0]])
    h = 1.0
    kernel, grad_kernel = rbf_kernel(particles, bandwidth=h)
    # off-diagonal (0, 1): x_j - x_i = particles[1] - particles[0] = +1
    expected = (2.0 / h) * (particles[1, 0] - particles[0, 0]) * kernel[0, 1]
    assert torch.allclose(grad_kernel[0, 1, 0], expected, atol=1e-6)
