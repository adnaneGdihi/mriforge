"""Tests for ``ConcomitantPhaseResidualLoss``.

Targets ``spectramr.models.losses.concomitant_phase_residual_loss``. CAUR /
idea 1 §1.2 residual ridge penalty.

Categories:

- Construction: invalid weight / reduction rejected
- Identity (zero residual) → loss = 0
- Loss grows quadratically with residual scale
- Reduction modes: mean / sum
- Gradient flows back to the residual tensor
- Registry: ``concomitant_phase_residual`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.concomitant_phase_residual_loss import (
    ConcomitantPhaseResidualLoss,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_weight_rejected() -> None:
    """``weight < 0`` raises."""
    with pytest.raises(ValueError, match="weight"):
        ConcomitantPhaseResidualLoss(weight=-0.1)


def test_invalid_reduction_rejected() -> None:
    """Only 'mean' and 'sum' are accepted."""
    with pytest.raises(ValueError, match="reduction"):
        ConcomitantPhaseResidualLoss(reduction="median")


# ---------------------------------------------------------------------------
# Closed-form
# ---------------------------------------------------------------------------


def test_zero_residual_yields_zero_loss() -> None:
    """``Δφ = 0`` → ``L = 0``."""
    loss = ConcomitantPhaseResidualLoss(weight=2.0)
    assert pytest.approx(loss(torch.zeros(2, 1, 4, 4)).item(), abs=1e-7) == 0.0


def test_loss_quadratic_in_residual_scale() -> None:
    """Doubling the residual quadruples the mean-squared loss."""
    loss = ConcomitantPhaseResidualLoss(weight=1.0)
    base = torch.randn(1, 1, 4, 4)
    val_1 = loss(base).item()
    val_2 = loss(2 * base).item()
    assert pytest.approx(val_2) == 4.0 * val_1


def test_weight_multiplies_loss() -> None:
    """Doubling the weight doubles the loss."""
    base = torch.randn(1, 1, 4, 4)
    val_1 = ConcomitantPhaseResidualLoss(weight=1.0)(base).item()
    val_3 = ConcomitantPhaseResidualLoss(weight=3.0)(base).item()
    assert pytest.approx(val_3) == 3.0 * val_1


# ---------------------------------------------------------------------------
# Reduction modes
# ---------------------------------------------------------------------------


def test_sum_reduction_scales_with_count() -> None:
    """``sum`` reduction yields ``N · mean``."""
    base = torch.full((1, 1, 4, 4), 0.5)
    n = base.numel()
    mean_loss = ConcomitantPhaseResidualLoss(reduction="mean")(base).item()
    sum_loss = ConcomitantPhaseResidualLoss(reduction="sum")(base).item()
    assert pytest.approx(sum_loss) == mean_loss * n


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_gradient_flows_to_residual() -> None:
    """Backward through the loss reaches the residual tensor."""
    loss = ConcomitantPhaseResidualLoss()
    residual = torch.randn(1, 1, 4, 4, requires_grad=True)
    loss(residual).backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


# ---------------------------------------------------------------------------
# Target argument is ignored
# ---------------------------------------------------------------------------


def test_target_argument_ignored() -> None:
    """Passing a target does not change the result."""
    base = torch.randn(1, 1, 4, 4)
    loss = ConcomitantPhaseResidualLoss(weight=1.5)
    val_no_target = loss(base).item()
    val_with_target = loss(base, target=torch.randn_like(base)).item()
    assert pytest.approx(val_no_target) == val_with_target


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``concomitant_phase_residual`` resolvable in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "concomitant_phase_residual" in list_available()
