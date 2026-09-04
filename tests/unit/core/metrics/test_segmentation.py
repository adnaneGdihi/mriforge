"""Tests for ``SegmentationEvaluator`` + ``dice_score``.

Targets ``spectramr.core.metrics.segmentation``. Wraps sklearn metrics
(``f1_score``, ``jaccard_score``, ``accuracy_score``) for binary and
multi-class segmentation evaluation.

Categories:

- ``dice_score`` returns 1.0 on perfect overlap, 0.0 on disjoint
- Binary evaluator returns ``dice``/``jaccard``/``precision``/``recall``/``f1``
- Multi-class evaluator returns ``*_macro`` variants
- ``argmax`` applied when prediction has channel dim
- ``ignore_index`` filters out marked pixels
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.core.metrics.segmentation import SegmentationEvaluator, dice_score


# ---------------------------------------------------------------------------
# dice_score (functional)
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap() -> None:
    """Identical masks → dice = 1.0."""
    mask = np.array([1, 1, 0, 0, 1])
    assert pytest.approx(dice_score(mask, mask)) == 1.0


def test_dice_disjoint_zero() -> None:
    """Disjoint binary masks → dice = 0."""
    a = np.array([1, 1, 0, 0])
    b = np.array([0, 0, 1, 1])
    assert pytest.approx(dice_score(a, b)) == 0.0


def test_dice_macro_average() -> None:
    """``average='macro'`` works on multi-class targets."""
    a = np.array([0, 1, 2, 0, 1, 2])
    b = a.copy()
    assert pytest.approx(dice_score(a, b, average="macro")) == 1.0


# ---------------------------------------------------------------------------
# Binary evaluator
# ---------------------------------------------------------------------------


def test_binary_evaluator_perfect_prediction() -> None:
    """All metrics = 1 when prediction matches target exactly."""
    pred = torch.tensor([[0, 1, 1], [0, 1, 0]])  # [B, H] for simplicity
    target = pred.clone()
    evaluator = SegmentationEvaluator(num_classes=2)
    out = evaluator(pred, target)
    assert pytest.approx(out["accuracy"]) == 1.0
    assert pytest.approx(out["dice"]) == 1.0
    assert pytest.approx(out["jaccard"]) == 1.0
    assert pytest.approx(out["f1"]) == 1.0


def test_binary_evaluator_returns_required_keys() -> None:
    """Binary evaluator returns the documented metric set."""
    pred = torch.tensor([[0, 1, 1, 0]])
    target = torch.tensor([[0, 1, 0, 0]])
    out = SegmentationEvaluator(num_classes=2)(pred, target)
    for k in ("accuracy", "precision", "recall", "f1", "dice", "jaccard"):
        assert k in out


def test_argmax_applied_for_4d_logits() -> None:
    """``[B, C, H, W]`` logits are reduced via argmax over channels."""
    # 2 classes; class-1 always wins.
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1] = 10.0
    target = torch.ones(1, 4, 4, dtype=torch.long)
    out = SegmentationEvaluator(num_classes=2)(logits, target)
    assert pytest.approx(out["accuracy"]) == 1.0


# ---------------------------------------------------------------------------
# Multi-class evaluator
# ---------------------------------------------------------------------------


def test_multi_class_returns_macro_variants() -> None:
    """``num_classes > 2`` switches to ``*_macro`` keys."""
    pred = torch.tensor([[0, 1, 2, 0, 1, 2]])
    target = pred.clone()
    out = SegmentationEvaluator(num_classes=3)(pred, target)
    for k in ("precision_macro", "recall_macro", "f1_macro", "dice_macro", "jaccard_macro"):
        assert k in out
    # No binary keys when multi-class.
    assert "dice" not in out


# ---------------------------------------------------------------------------
# ignore_index
# ---------------------------------------------------------------------------


def test_ignore_index_filters_pixels() -> None:
    """Pixels marked with ``ignore_index`` don't contribute to the metric."""
    # Target has -1 at positions where pred is wrong → ignored, accuracy = 1.
    pred = torch.tensor([[0, 1, 0, 1]])
    target = torch.tensor([[0, 1, -1, -1]])
    out = SegmentationEvaluator(num_classes=2, ignore_index=-1)(pred, target)
    assert pytest.approx(out["accuracy"]) == 1.0


# ---------------------------------------------------------------------------
# scikit-learn optional-dependency guard (6.1). Raises at USE time, keeping
# the module importable so walk-discovery still registers other metrics.
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_sklearn_flag() -> None:
    """The module imports even in a torch-only env and exposes the flag."""
    import spectramr.core.metrics.segmentation as seg

    assert isinstance(seg.SKLEARN_AVAILABLE, bool)


def test_dice_score_raises_clear_error_when_sklearn_absent(monkeypatch) -> None:
    """``dice_score`` raises a clear ImportError when sklearn is unavailable."""
    import spectramr.core.metrics.segmentation as seg

    monkeypatch.setattr(seg, "SKLEARN_AVAILABLE", False)
    with pytest.raises(ImportError, match="scikit-learn"):
        seg.dice_score(np.array([1, 0]), np.array([1, 0]))


def test_evaluator_raises_clear_error_when_sklearn_absent(monkeypatch) -> None:
    """The evaluator raises a clear ImportError when sklearn is unavailable."""
    import spectramr.core.metrics.segmentation as seg

    monkeypatch.setattr(seg, "SKLEARN_AVAILABLE", False)
    with pytest.raises(ImportError, match="scikit-learn"):
        SegmentationEvaluator(num_classes=2)(
            torch.tensor([[0, 1, 0, 1]]), torch.tensor([[0, 1, 0, 1]])
        )
