"""Tests for ``DeepSupervisionLoss``.

Targets ``mriforge.models.losses.deep_supervision_loss``. Computes a base
loss at multiple intermediate layer outputs to improve gradient flow,
auto-resizing target to match each intermediate's spatial dims.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.models.losses.deep_supervision_loss import DeepSupervisionLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default base loss is ``L1Loss`` with weight 0.1."""
    loss = DeepSupervisionLoss()
    assert isinstance(loss.base_loss_fn, nn.L1Loss)
    assert loss.weight == 0.1


def test_explicit_base_loss_and_weight() -> None:
    """Custom base loss / weight are persisted."""
    loss = DeepSupervisionLoss(base_loss_fn=nn.MSELoss(), weight=0.5)
    assert isinstance(loss.base_loss_fn, nn.MSELoss)
    assert loss.weight == 0.5


# ---------------------------------------------------------------------------
# Empty list
# ---------------------------------------------------------------------------


def test_empty_intermediate_returns_zero() -> None:
    """No intermediates → loss = 0."""
    loss = DeepSupervisionLoss()
    target = torch.randn(1, 1, 16, 16)
    out = loss([], target)
    assert out.item() == 0.0


# ---------------------------------------------------------------------------
# Single intermediate matching target shape
# ---------------------------------------------------------------------------


def test_single_intermediate_matching_shape_zero_residual() -> None:
    """``loss([x], x)`` is zero when ``base_loss(x, x) == 0`` (modulo ε)."""
    loss = DeepSupervisionLoss(weight=1.0)
    x = torch.randn(1, 1, 16, 16)
    out = loss([x], x)
    assert abs(out.item()) < 1e-6


def test_loss_scaled_by_weight() -> None:
    """Doubling weight doubles the loss."""
    target = torch.zeros(1, 1, 16, 16)
    pred = torch.full_like(target, 1.0)
    out_w1 = DeepSupervisionLoss(weight=1.0)([pred], target)
    out_w2 = DeepSupervisionLoss(weight=2.0)([pred], target)
    assert pytest.approx(out_w2.item(), rel=1e-5) == 2.0 * out_w1.item()


# ---------------------------------------------------------------------------
# Multi-level: target gets resized
# ---------------------------------------------------------------------------


def test_multi_level_with_size_mismatch_resizes_target() -> None:
    """Intermediate at lower resolution → target is auto-downsampled."""
    loss = DeepSupervisionLoss(weight=1.0)
    target = torch.randn(1, 1, 32, 32)
    # Intermediate is at half resolution.
    intermediate_8 = torch.randn(1, 1, 16, 16)
    intermediate_4 = torch.randn(1, 1, 8, 8)
    out = loss([intermediate_8, intermediate_4], target)
    assert torch.is_tensor(out)
    assert out.dim() == 0
    assert out.item() >= 0


def test_loss_accumulates_across_levels() -> None:
    """Total loss = sum over levels (then × weight)."""
    target = torch.zeros(1, 1, 16, 16)
    pred1 = torch.full_like(target, 1.0)
    pred2 = torch.full_like(target, 2.0)
    out_single = DeepSupervisionLoss(weight=1.0)([pred1], target).item()
    out_two = DeepSupervisionLoss(weight=1.0)([pred1, pred2], target).item()
    # With L1 loss: |1| = 1, |2| = 2 → total = 3.
    assert pytest.approx(out_two, abs=1e-4) == out_single + 2.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_deep_supervision_registered() -> None:
    """``deep_supervision`` is in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "deep_supervision" in list_available()


# ---------------------------------------------------------------------------
# Gradient flow (regression: previously accumulated via ``.item()``)
# ---------------------------------------------------------------------------


def test_supervision_propagates_gradient_to_every_level() -> None:
    """The summed loss must carry gradient back to each intermediate output.

    Pre-fix ``total_loss += level_loss.item()`` detached every level, so the
    deep-supervision signal contributed zero gradient.
    """
    loss = DeepSupervisionLoss(base_loss_fn=nn.MSELoss(), weight=1.0)
    target = torch.randn(1, 1, 16, 16)
    intermediates = [
        torch.randn(1, 1, 16, 16, requires_grad=True),
        torch.randn(1, 1, 8, 8, requires_grad=True),
    ]

    out = loss(intermediates, target)
    assert out.requires_grad and out.grad_fn is not None
    out.backward()
    for level in intermediates:
        assert level.grad is not None and level.grad.abs().sum() > 0
