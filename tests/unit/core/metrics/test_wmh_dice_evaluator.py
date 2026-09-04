"""Unit tests for the WMH downstream-task evaluator scaffold."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.wmh_dice_evaluator import (  # noqa: E402
    LazySegmenterAdapter,
    WMHDiceEvaluator,
)


def test_dice_perfect_match() -> None:
    evaluator = WMHDiceEvaluator()
    vol = torch.zeros(1, 1, 4, 8, 8)
    vol[..., :, 2:6, 2:6] = 1.0  # constant bright region
    score = evaluator(vol, vol)
    assert torch.isclose(score, torch.tensor([1.0]), atol=1e-3)


def test_dice_identity_with_mask() -> None:
    evaluator = WMHDiceEvaluator()
    vol = torch.rand(1, 1, 4, 8, 8)
    mask = torch.ones(1, 1, 4, 8, 8)
    score = evaluator(vol, vol, mask=mask)
    assert torch.allclose(score, torch.tensor([1.0]), atol=1e-3)


def test_dice_disjoint_predictions_low() -> None:
    """Pred bright in one corner, target bright in the opposite corner."""
    evaluator = WMHDiceEvaluator()
    pred = torch.zeros(1, 1, 4, 8, 8)
    target = torch.zeros(1, 1, 4, 8, 8)
    pred[..., :, :3, :3] = 1.0
    target[..., :, 5:, 5:] = 1.0
    score = evaluator(pred, target).item()
    assert score < 0.2


def test_lazy_segmenter_rejects_bad_percentile() -> None:
    with pytest.raises(ValueError, match="percentile"):
        LazySegmenterAdapter(percentile=1.5)


def test_evaluator_rejects_shape_mismatch() -> None:
    evaluator = WMHDiceEvaluator()
    pred = torch.zeros(1, 1, 4, 8, 8)
    target = torch.zeros(1, 1, 4, 8, 16)
    with pytest.raises(ValueError, match="match shapes"):
        evaluator(pred, target)


def test_evaluator_rejects_wrong_ndim() -> None:
    evaluator = WMHDiceEvaluator()
    pred = torch.zeros(1, 1, 8, 8)
    target = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match=r"\(B, C, D, H, W\)"):
        evaluator(pred, target)
