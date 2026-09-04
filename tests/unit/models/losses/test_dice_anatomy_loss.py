"""Tests for ``SegmentationDiceLoss``.

Targets ``spectramr.models.losses.dice_anatomy_loss``. Anti-hallucination
loss that runs a frozen segmenter on both prediction and target and
penalises class-wise Dice disagreement.

Categories:

- Construction: ``smooth > 0``, ``require_segmenter`` toggles
- Missing segmenter raises (require=True) / returns zero (require=False)
- Identical inputs through a deterministic segmenter → loss ≈ 0
- ``_soft_dice_per_class`` helper: identity returns 1, disjoint sets returns 0
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.dice_anatomy_loss import (
    SegmentationDiceLoss,
    _soft_dice_per_class,
)


# ---------------------------------------------------------------------------
# _soft_dice_per_class helper
# ---------------------------------------------------------------------------


def test_soft_dice_identity_returns_one() -> None:
    """``dice(p, p)`` returns 1.0 per class (perfect overlap)."""
    p = torch.rand(2, 3, 8, 8)
    out = _soft_dice_per_class(p, p, smooth=1e-6)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-3)


def test_soft_dice_disjoint_returns_zero() -> None:
    """Disjoint masks → Dice ≈ 0."""
    p = torch.zeros(1, 2, 4, 4)
    t = torch.zeros(1, 2, 4, 4)
    p[:, 0, :2] = 1.0  # class 0 in top half
    t[:, 0, 2:] = 1.0  # class 0 in bottom half (disjoint)
    out = _soft_dice_per_class(p, t, smooth=1e-6)
    assert out[0].item() < 0.01


def test_soft_dice_smooth_avoids_zero_division() -> None:
    """Both inputs zero → smooth keeps result finite."""
    p = torch.zeros(1, 2, 4, 4)
    out = _soft_dice_per_class(p, p, smooth=1.0)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_persists_attributes() -> None:
    """Constructor stores ``smooth``, ``ignore_background``, ``require_segmenter``."""
    loss = SegmentationDiceLoss(smooth=1e-3, ignore_background=False, require_segmenter=False)
    assert loss.smooth == 1e-3
    assert loss.ignore_background is False
    assert loss.require_segmenter is False


def test_invalid_smooth_raises() -> None:
    """``smooth <= 0`` → ``ValueError``."""
    with pytest.raises(ValueError, match="smooth must be > 0"):
        SegmentationDiceLoss(smooth=0.0)
    with pytest.raises(ValueError, match="smooth must be > 0"):
        SegmentationDiceLoss(smooth=-1e-4)


# ---------------------------------------------------------------------------
# Missing segmenter behaviour
# ---------------------------------------------------------------------------


def test_missing_segmenter_with_require_raises() -> None:
    """``require_segmenter=True`` (default) → ``ValueError`` when no segmenter."""
    loss = SegmentationDiceLoss()
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)
    with pytest.raises(ValueError, match="requires context\\['segmenter'\\]"):
        loss(pred, target, context={})


def test_missing_segmenter_with_no_require_returns_zero() -> None:
    """``require_segmenter=False`` → returns scalar zero."""
    loss = SegmentationDiceLoss(require_segmenter=False)
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)
    out = loss(pred, target, context={})
    assert out.dim() == 0
    assert out.item() == 0.0


def test_non_callable_segmenter_raises() -> None:
    """A non-callable ``segmenter`` → ``TypeError``."""
    loss = SegmentationDiceLoss()
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8)
    with pytest.raises(TypeError, match="must be callable"):
        loss(pred, target, context={"segmenter": "not_callable"})


# ---------------------------------------------------------------------------
# Forward with a fake segmenter
# ---------------------------------------------------------------------------


class _DummySegmenter(nn.Module):
    """Deterministic linear segmenter for testing."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, num_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_identical_inputs_yield_near_zero_loss() -> None:
    """Same input image → segmenter outputs match → Dice = 1 → loss ≈ 0."""
    loss_fn = SegmentationDiceLoss(require_segmenter=True)
    seg = _DummySegmenter(num_classes=3).eval()
    x = torch.randn(1, 1, 8, 8)
    out = loss_fn(x, x, context={"segmenter": seg})
    assert out.item() < 0.05


def test_different_inputs_yield_positive_loss() -> None:
    """Different inputs through the segmenter → positive Dice loss."""
    torch.manual_seed(0)
    loss_fn = SegmentationDiceLoss(require_segmenter=True)
    seg = _DummySegmenter(num_classes=3).eval()
    pred = torch.randn(1, 1, 8, 8)
    target = torch.randn(1, 1, 8, 8) * 5  # very different
    out = loss_fn(pred, target, context={"segmenter": seg})
    assert out.item() > 0


def test_ignore_background_reduces_average() -> None:
    """``ignore_background=True`` drops class 0 from the averaging."""
    seg = _DummySegmenter(num_classes=4).eval()
    x = torch.randn(1, 1, 8, 8)
    loss_with_bg = SegmentationDiceLoss(ignore_background=False)(
        x, x, context={"segmenter": seg}
    ).item()
    loss_no_bg = SegmentationDiceLoss(ignore_background=True)(
        x, x, context={"segmenter": seg}
    ).item()
    # Both should be near zero for x == x — sanity that no exception raised.
    assert loss_with_bg < 0.05
    assert loss_no_bg < 0.05


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_segmentation_dice_registered() -> None:
    """``segmentation_dice`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "segmentation_dice" in list_available()
