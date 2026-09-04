"""Tests for tropical geometry MRF / quantitative-map primitives."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.tropical_geometry import (
    TropicalRationalMap,
    active_cone_indices,
    log_sum_exp_tropical,
    tropical_polynomial,
)


def test_tropical_polynomial_basic_max_plus() -> None:
    """Verify p(r) = max_i (a_i^T r + b_i) on a known small example."""
    # Three lines through (0, ±1) and the constant 0.
    slopes = torch.tensor([[1.0], [-1.0], [0.0]])
    intercepts = torch.tensor([0.0, 0.0, 1.0])
    points = torch.tensor([[-2.0], [-0.5], [0.0], [0.5], [2.0]])
    out = tropical_polynomial(points, slopes, intercepts)
    expected = torch.tensor([2.0, 1.0, 1.0, 1.0, 2.0])
    assert torch.allclose(out, expected)


def test_log_sum_exp_tropical_converges_to_max_at_large_beta() -> None:
    """Soft tropical polynomial → hard tropical polynomial as β → ∞."""
    torch.manual_seed(0)
    slopes = torch.randn(5, 3)
    intercepts = torch.randn(5)
    points = torch.randn(32, 3)
    hard = tropical_polynomial(points, slopes, intercepts)
    soft = log_sum_exp_tropical(points, slopes, intercepts, beta=64.0)
    # As β → ∞ the soft version is bounded between max and max + (log K)/β.
    delta = (soft - hard).abs().max().item()
    assert delta < 0.1


def test_active_cone_classifier_partitions_input() -> None:
    """argmax of monomials partitions the input space (cone classifier σ_n)."""
    slopes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    intercepts = torch.zeros(3)
    points = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]])
    idx = active_cone_indices(points, slopes, intercepts)
    assert torch.equal(idx, torch.tensor([0, 1, 2]))


def test_tropical_rational_map_shape_and_grad() -> None:
    """The difference-of-tropical-polynomials net is differentiable."""
    net = TropicalRationalMap(in_dim=3, out_dim=2, n_monomials_pos=8, n_monomials_neg=4)
    points = torch.randn(16, 3, requires_grad=True)
    out = net(points)
    assert out.shape == (16, 2)
    out.sum().backward()
    assert points.grad is not None
    assert net.slopes_pos.grad is not None
    assert net.slopes_neg.grad is not None


def test_tropical_rational_map_is_continuous() -> None:
    """Continuous piecewise-affine: small step in input → small step in output."""
    net = TropicalRationalMap(in_dim=2, out_dim=1)
    base = torch.zeros(1, 2)
    perturbation = torch.tensor([[1e-4, 0.0]])
    out_a = net(base)
    out_b = net(base + perturbation)
    assert (out_a - out_b).abs().max().item() < 1e-1


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        TropicalRationalMap(in_dim=0, out_dim=1)
    with pytest.raises(ValueError):
        TropicalRationalMap(in_dim=3, out_dim=1, n_monomials_pos=0)
    with pytest.raises(ValueError):
        TropicalRationalMap(in_dim=3, out_dim=1, beta=-1.0)
    with pytest.raises(ValueError):
        log_sum_exp_tropical(torch.zeros(3, 2), torch.zeros(4, 2), torch.zeros(4), beta=0.0)
    with pytest.raises(ValueError):
        tropical_polynomial(torch.zeros(3, 2), torch.zeros(4), torch.zeros(4))  # 1-D slopes
