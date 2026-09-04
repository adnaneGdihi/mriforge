"""Tests for ``MarkerCorruptionLoss``.

Targets ``spectramr.models.losses.marker_corruption_loss``. Measures L1
fidelity inside a known marker mask + smoothness across the marker /
anatomy boundary.

Categories:

- Construction: defaults match the docstring weights
- Without marker info → falls back to plain L1 against the target
- Identity (pred == ideal == target) → loss = 0 inside mask
- Disabling consistency (``lambda_consistency = 0``) yields a pure
  marker-region L1
- Mask is normalised by area (zero-area mask is safe)
- Complex ``ideal_marker`` is auto-stacked to real
- Gradient flows back to the prediction
- Registry: ``marker_corruption`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.marker_corruption_loss import MarkerCorruptionLoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_weights() -> None:
    """Default ``lambda_l1=1.0``, ``lambda_consistency=0.1``."""
    loss = MarkerCorruptionLoss()
    assert loss.lambda_l1 == 1.0
    assert loss.lambda_consistency == 0.1


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


def test_no_mask_falls_back_to_plain_l1() -> None:
    """Without ``marker_mask`` / ``ideal_marker``, the loss is plain L1."""
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 1, 4, 4), 1.0)
    loss = MarkerCorruptionLoss()
    val = loss(pred, target).item()
    # Plain L1 = mean(|0 - 1|) = 1
    assert pytest.approx(val) == 1.0


# ---------------------------------------------------------------------------
# Marker-region L1 (disable consistency for a clean closed form)
# ---------------------------------------------------------------------------


def test_zero_loss_when_pred_matches_ideal_in_mask() -> None:
    """Inside the mask, ``pred == ideal`` → marker-L1 component is zero."""
    pred = torch.full((1, 1, 4, 4), 2.0)
    ideal = torch.full((1, 1, 4, 4), 2.0)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :2] = 1.0  # 4 marker voxels in a 16-pixel image
    loss = MarkerCorruptionLoss(lambda_l1=1.0, lambda_consistency=0.0)
    val = loss(
        pred,
        target=torch.zeros_like(pred),  # ignored when mask is provided
        marker_mask=mask,
        ideal_marker=ideal,
    ).item()
    assert pytest.approx(val, abs=1e-7) == 0.0


def test_marker_l1_normalised_by_mask_area() -> None:
    """``L1_marker = sum(|pred - ideal| · mask) / sum(mask)``."""
    pred = torch.full((1, 1, 4, 4), 1.0)
    ideal = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :2] = 1.0  # 4 voxels
    loss = MarkerCorruptionLoss(lambda_l1=1.0, lambda_consistency=0.0)
    val = loss(
        pred,
        target=torch.zeros_like(pred),
        marker_mask=mask,
        ideal_marker=ideal,
    ).item()
    # |1 - 0| · 4 / 4 = 1
    assert pytest.approx(val) == 1.0


def test_marker_l1_grows_with_pred_deviation() -> None:
    """Larger pred-vs-ideal gap → larger marker-L1."""
    ideal = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :2] = 1.0
    loss = MarkerCorruptionLoss(lambda_l1=1.0, lambda_consistency=0.0)

    val_small = loss(
        torch.full((1, 1, 4, 4), 1.0),
        target=torch.zeros(1, 1, 4, 4),
        marker_mask=mask,
        ideal_marker=ideal,
    ).item()
    val_big = loss(
        torch.full((1, 1, 4, 4), 5.0),
        target=torch.zeros(1, 1, 4, 4),
        marker_mask=mask,
        ideal_marker=ideal,
    ).item()
    assert val_big > val_small


# ---------------------------------------------------------------------------
# Complex ideal handling
# ---------------------------------------------------------------------------


def test_complex_ideal_auto_stacked() -> None:
    """A complex ``ideal_marker`` is internally split to real/imag stacks."""
    pred = torch.zeros(1, 2, 4, 4)  # 2-channel real
    ideal = torch.complex(
        torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)
    )
    mask = torch.ones(1, 1, 4, 4)
    loss = MarkerCorruptionLoss(lambda_l1=1.0, lambda_consistency=0.0)
    val = loss(
        pred, target=torch.zeros_like(pred),
        marker_mask=mask, ideal_marker=ideal,
    ).item()
    # pred and ideal are both zero → marker-L1 ≈ 0 (a small ε is allowed
    # because the complex→real stack picks up float32 ε noise).
    assert pytest.approx(val, abs=1e-5) == 0.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_prediction() -> None:
    """Backward reaches the prediction tensor."""
    pred = torch.randn(1, 1, 4, 4, requires_grad=True)
    ideal = torch.randn(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    loss = MarkerCorruptionLoss()
    loss(pred, target=torch.zeros_like(pred), marker_mask=mask, ideal_marker=ideal).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``marker_corruption`` resolvable."""
    from spectramr.models.losses.registry import list_available

    assert "marker_corruption" in list_available()
