"""Tests for ``CharbonnierLoss``.

Targets ``mriforge.models.losses.charbonnier_loss``. The Charbonnier penalty
is a smooth-L1 surrogate: ``sqrt((pred - target)² + ε²)``. It approaches
L1 for large residuals and L2 for small ones — bounded gradient at zero.
Standard pixel-loss for SR / unrolled MRI reconstruction.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.losses.charbonnier_loss import CharbonnierLoss


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_epsilon_raises() -> None:
    """ε ≤ 0 fails loud — the smoothing constant must be positive."""
    with pytest.raises(ValueError, match="epsilon"):
        CharbonnierLoss(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        CharbonnierLoss(epsilon=-1e-3)


def test_invalid_reduction_raises() -> None:
    """Reduction other than mean/sum/none fails loud (CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="reduction"):
        CharbonnierLoss(reduction="median")


# ---------------------------------------------------------------------------
# Identity / zero residual
# ---------------------------------------------------------------------------


def test_loss_at_zero_residual_equals_epsilon() -> None:
    """When pred == target, per-voxel value is exactly ε."""
    loss_fn = CharbonnierLoss(epsilon=1e-3, reduction="mean")
    x = torch.randn(2, 1, 8, 8)
    out = loss_fn(x, x)
    assert math.isclose(out.item(), 1e-3, rel_tol=1e-6)


def test_loss_is_non_negative() -> None:
    """Charbonnier penalty is always ≥ 0 (sqrt of non-negative quantity)."""
    loss_fn = CharbonnierLoss()
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)
    out = loss_fn(pred, target)
    assert out.item() >= 0


def test_loss_grows_monotonically_with_residual() -> None:
    """A larger residual produces a larger loss."""
    loss_fn = CharbonnierLoss(epsilon=1e-3)
    target = torch.zeros(1, 1, 4, 4)
    out_small = loss_fn(torch.full_like(target, 0.1), target)
    out_large = loss_fn(torch.full_like(target, 1.0), target)
    assert out_large > out_small


def test_loss_approaches_l1_for_large_residuals() -> None:
    """For r >> ε: sqrt(r² + ε²) ≈ |r| → loss ≈ |r|."""
    loss_fn = CharbonnierLoss(epsilon=1e-6, reduction="mean")
    target = torch.zeros(1, 1, 16, 16)
    pred = torch.full_like(target, 5.0)  # |residual| = 5, ε << 5
    out = loss_fn(pred, target)
    # Tolerance: ε² / (2|r|) error in Taylor expansion.
    assert abs(out.item() - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Reduction modes
# ---------------------------------------------------------------------------


def test_reduction_mean_returns_scalar() -> None:
    """``reduction='mean'`` → 0-dim scalar tensor."""
    loss_fn = CharbonnierLoss(reduction="mean")
    out = loss_fn(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4))
    assert out.dim() == 0


def test_reduction_sum_returns_scalar() -> None:
    """``reduction='sum'`` → 0-dim scalar tensor."""
    loss_fn = CharbonnierLoss(reduction="sum")
    out = loss_fn(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4))
    assert out.dim() == 0


def test_reduction_none_returns_per_voxel_tensor() -> None:
    """``reduction='none'`` → tensor matching input shape."""
    loss_fn = CharbonnierLoss(reduction="none")
    pred = torch.randn(2, 1, 4, 4)
    target = torch.randn(2, 1, 4, 4)
    out = loss_fn(pred, target)
    assert out.shape == pred.shape


def test_sum_equals_mean_times_numel() -> None:
    """``sum`` reduction equals ``mean`` × number of elements."""
    pred = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 4, 4)
    mean_loss = CharbonnierLoss(reduction="mean")(pred, target)
    sum_loss = CharbonnierLoss(reduction="sum")(pred, target)
    expected_sum = mean_loss * pred.numel()
    assert torch.allclose(sum_loss, expected_sum, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_prediction() -> None:
    """Backward through the loss reaches the prediction tensor."""
    loss_fn = CharbonnierLoss()
    pred = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.zeros(2, 1, 4, 4)
    loss = loss_fn(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_gradient_bounded_at_zero_residual() -> None:
    """At zero residual, the gradient is finite (no division explosion)."""
    loss_fn = CharbonnierLoss(epsilon=1e-3)
    pred = torch.zeros(2, 1, 4, 4, requires_grad=True)
    target = torch.zeros(2, 1, 4, 4)
    loss = loss_fn(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 8), (2, 1, 16, 16), (1, 2, 32, 32), (4, 1, 64, 64), (1, 1, 16, 16, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_shape_matrix(shape: tuple[int, ...]) -> None:
    """Loss accepts arbitrary tensor shapes (domain-agnostic per docstring)."""
    loss_fn = CharbonnierLoss()
    pred = torch.randn(*shape)
    target = torch.randn(*shape)
    out = loss_fn(pred, target)
    assert torch.isfinite(out)


def test_loss_registered_under_alias() -> None:
    """The ``@register_loss`` decorator registers the canonical name."""
    from mriforge.models.losses.registry import list_available

    available = list_available()
    assert "charbonnier" in available, (
        f"'charbonnier' not in registry; available: {available[:10]}"
    )
