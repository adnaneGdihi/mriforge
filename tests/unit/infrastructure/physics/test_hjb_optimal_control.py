"""Tests for HJB optimal-control primitives."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.hjb_optimal_control import (
    ValueFunctionMLP,
    closed_loop_control,
    hjb_residual_loss,
    pilot_gradient_flow,
    pilot_limit_reduction,
)


def _running_cost(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Quadratic state-cost test fixture."""
    _ = time
    return (state ** 2).sum(dim=-1)


def test_value_function_forward_shape() -> None:
    net = ValueFunctionMLP(state_dim=3)
    state = torch.randn(8, 3)
    time = torch.rand(8)
    v = net(state, time)
    assert v.shape == (8,)


def test_closed_loop_control_has_unit_norm_scaled_by_g_max() -> None:
    """||u*||₂ = g_max for any non-degenerate ∇V."""
    net = ValueFunctionMLP(state_dim=3)
    state = torch.randn(8, 3)
    time = torch.rand(8)
    u = closed_loop_control(net, state, time, g_max=2.5)
    norms = u.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 2.5), atol=1e-4)


def test_hjb_residual_is_nonnegative_scalar() -> None:
    """The PDE residual is the squared L² norm — non-negative scalar."""
    net = ValueFunctionMLP(state_dim=3)
    state = torch.randn(8, 3, requires_grad=True)
    time = torch.rand(8, requires_grad=True)
    loss = hjb_residual_loss(net, state, time, _running_cost, g_max=1.0)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_pilot_limit_returns_unit_norm_direction() -> None:
    """As G_max → ∞ the closed-loop direction matches PILOT's gradient direction."""
    net = ValueFunctionMLP(state_dim=3)
    state = torch.randn(8, 3)
    time = torch.rand(8)
    direction = pilot_limit_reduction(net, state, time, g_max=1e6)
    norms = direction.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_pilot_gradient_flow_for_quadratic_cost() -> None:
    """For L=||k||², ∇L = 2k → -∇L direction = -k/||k||."""
    state = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    time = torch.zeros(2)
    direction = pilot_gradient_flow(_running_cost, state, time)
    expected = -state / state.norm(dim=-1, keepdim=True)
    assert torch.allclose(direction, expected, atol=1e-5)


def test_value_function_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        ValueFunctionMLP(state_dim=0)
    net = ValueFunctionMLP(state_dim=3)
    with pytest.raises(ValueError):
        net(torch.zeros(3), torch.zeros(3))  # state not 2-D
    with pytest.raises(ValueError):
        net(torch.zeros(3, 3), torch.zeros(2))  # batch mismatch


def test_hjb_residual_value_fn_must_return_1d() -> None:
    def bad_value_fn(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(s.shape[0], 2)

    with pytest.raises(ValueError):
        hjb_residual_loss(bad_value_fn, torch.zeros(4, 3), torch.zeros(4), _running_cost)
