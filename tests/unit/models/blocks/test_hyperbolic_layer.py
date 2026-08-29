"""Tests for Poincaré-ball hyperbolic operations."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.hyperbolic_layer import (
    PoincareLayer,
    exp_map_zero,
    hyperbolic_tree_distance_loss,
    log_map_zero,
    mobius_add,
    poincare_distance,
)


def test_exp_log_are_mutual_inverses_at_origin() -> None:
    """log_0(exp_0(v)) = v for tangent vectors v at the origin."""
    torch.manual_seed(0)
    v = torch.randn(8, 4) * 0.1  # small magnitudes to stay inside the ball
    z = exp_map_zero(v, c=1.0)
    v_back = log_map_zero(z, c=1.0)
    assert torch.allclose(v, v_back, atol=1e-4)


def test_exp_map_stays_in_ball() -> None:
    """For any v, ||exp_0(v)|| ≤ 1/sqrt(c) = 1 with c=1 (saturating at large
    norms due to tanh).
    """
    v = torch.randn(64, 8) * 5.0  # large vectors
    z = exp_map_zero(v, c=1.0)
    norms = z.norm(dim=-1)
    assert (norms <= 1.0 + 1e-5).all()


def test_zero_curvature_recovers_euclidean() -> None:
    """At c=0 every Möbius op is the Euclidean one."""
    x = torch.randn(4, 3)
    y = torch.randn(4, 3)
    assert torch.allclose(exp_map_zero(x, c=0.0), x)
    assert torch.allclose(log_map_zero(x, c=0.0), x)
    assert torch.allclose(mobius_add(x, y, c=0.0), x + y)
    eucl = (x - y).norm(dim=-1)
    hyp = poincare_distance(x, y, c=0.0)
    assert torch.allclose(eucl, hyp)


def test_mobius_add_identity_with_origin() -> None:
    """x ⊕_c 0 = x and 0 ⊕_c y = y."""
    x = torch.randn(4, 3) * 0.1
    zeros = torch.zeros_like(x)
    assert torch.allclose(mobius_add(x, zeros, c=1.0), x)
    assert torch.allclose(mobius_add(zeros, x, c=1.0), x)


def test_poincare_distance_symmetric_and_nonneg() -> None:
    torch.manual_seed(0)
    u = torch.randn(8, 4) * 0.1
    v = torch.randn(8, 4) * 0.1
    d_uv = poincare_distance(u, v, c=1.0)
    d_vu = poincare_distance(v, u, c=1.0)
    assert torch.allclose(d_uv, d_vu, atol=1e-5)
    assert (d_uv >= 0.0).all()


def test_poincare_layer_forward_gradient_flow() -> None:
    layer = PoincareLayer(in_features=8, out_features=4, curvature=1.0)
    x = torch.randn(16, 8, requires_grad=True)
    z = layer(x)
    assert z.shape == (16, 4)
    z.sum().backward()
    assert x.grad is not None
    assert layer.linear.weight.grad is not None
    assert layer.bias.grad is not None


def test_tree_distance_loss_zero_when_siblings_collocate() -> None:
    embeddings = torch.zeros(5, 4)  # all at origin → all distances 0
    mask = torch.eye(5).bool() | torch.ones(5, 5).bool()  # all-siblings
    loss = hyperbolic_tree_distance_loss(embeddings, mask, c=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_tree_distance_loss_positive_on_disparate_siblings() -> None:
    embeddings = torch.tensor([[0.1, 0.0], [-0.1, 0.0]])
    mask = torch.tensor([[True, True], [True, True]])
    loss = hyperbolic_tree_distance_loss(embeddings, mask, c=1.0)
    assert loss.item() > 0.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        exp_map_zero(torch.zeros(3), c=-1.0)
    with pytest.raises(ValueError):
        PoincareLayer(in_features=4, out_features=2, curvature=-0.1)
    with pytest.raises(ValueError):
        hyperbolic_tree_distance_loss(torch.zeros(4), torch.zeros(4, 4).bool())
    with pytest.raises(ValueError):
        hyperbolic_tree_distance_loss(torch.zeros(4, 2), torch.zeros(3, 3).bool())
