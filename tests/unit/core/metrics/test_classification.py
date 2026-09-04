"""Tests for ``ClassificationEvaluator``.

Targets ``spectramr.core.metrics.classification``. Wraps scikit-learn's
``accuracy``, ``precision``, ``recall``, ``f1`` scores with the project's
unified pred/target signature.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.classification import ClassificationEvaluator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_average_is_macro() -> None:
    """Default ``average`` is ``'macro'``."""
    e = ClassificationEvaluator()
    assert e.average == "macro"


def test_invalid_average_raises_not_silent_fallback() -> None:
    """Unknown average string now RAISES (no silent coercion to ``'macro'``)."""
    with pytest.raises(ValueError):
        ClassificationEvaluator(average="not_a_real_average")


@pytest.mark.parametrize("average", ["micro", "macro", "weighted", "binary"])
def test_valid_averages_are_persisted(average: str) -> None:
    """All four scikit-learn averages are preserved verbatim."""
    e = ClassificationEvaluator(average=average)
    assert e.average == average


# ---------------------------------------------------------------------------
# __call__ — perfect prediction
# ---------------------------------------------------------------------------


def test_perfect_prediction_yields_one_for_all_metrics_binary() -> None:
    """Identical pred/target → all metrics = 1.0 (binary)."""
    e = ClassificationEvaluator(num_classes=2)
    pred = torch.tensor([0, 1, 0, 1, 1])
    target = pred.clone()
    out = e(pred, target)
    assert out["accuracy"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0


def test_perfect_prediction_multiclass() -> None:
    """Identical pred/target → all metrics = 1.0 (multiclass)."""
    e = ClassificationEvaluator(num_classes=3, average="macro")
    pred = torch.tensor([0, 1, 2, 0, 1, 2])
    target = pred.clone()
    out = e(pred, target)
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0


# ---------------------------------------------------------------------------
# __call__ — chance / wrong prediction
# ---------------------------------------------------------------------------


def test_all_wrong_yields_zero_accuracy() -> None:
    """Inverted predictions → accuracy = 0."""
    e = ClassificationEvaluator(num_classes=2)
    pred = torch.tensor([0, 0, 0, 0])
    target = torch.tensor([1, 1, 1, 1])
    out = e(pred, target)
    assert out["accuracy"] == 0.0


def test_zero_division_safe() -> None:
    """``zero_division=0`` ensures no division-by-zero exception."""
    e = ClassificationEvaluator(num_classes=2)
    # Pred all 0s, target all 0s → recall is undefined for class 1; sklearn returns 0
    pred = torch.tensor([0, 0, 0])
    target = torch.tensor([0, 0, 0])
    out = e(pred, target)  # must not raise
    assert "f1" in out


# ---------------------------------------------------------------------------
# __call__ — input shape
# ---------------------------------------------------------------------------


def test_call_accepts_logits_2d() -> None:
    """``[B, C]`` logits are argmax'd before scoring."""
    e = ClassificationEvaluator(num_classes=3, average="macro")
    logits = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.0, 0.0, 1.0]])
    target = torch.tensor([0, 1, 2])
    out = e(logits, target)
    assert out["accuracy"] == 1.0


def test_call_accepts_class_indices_1d() -> None:
    """``[B]`` class indices skip argmax."""
    e = ClassificationEvaluator(num_classes=2)
    pred = torch.tensor([0, 1])
    target = torch.tensor([0, 1])
    out = e(pred, target)
    assert out["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_keys() -> None:
    """Output dict contains accuracy, precision, recall, f1."""
    e = ClassificationEvaluator(num_classes=2)
    out = e(torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert {"accuracy", "precision", "recall", "f1"} == out.keys()


def test_output_values_are_floats_in_unit_interval() -> None:
    """All metric values fall in ``[0, 1]``."""
    e = ClassificationEvaluator(num_classes=2)
    out = e(torch.tensor([0, 1, 1, 0]), torch.tensor([0, 1, 0, 1]))
    for val in out.values():
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Regression: no-silent-fallback on unknown average (CLAUDE.md #9)
# ---------------------------------------------------------------------------


def test_unknown_average_raises() -> None:
    """An unknown ``average`` must raise ``ValueError`` (was silently → 'macro').

    Regression for the no-silent-fallback fix: a typo'd averaging strategy used
    to degrade to ``'macro'`` and report a different, plausible-looking score.
    It must now fail loudly at construction time.
    """
    with pytest.raises(ValueError):
        ClassificationEvaluator(average="bogus")


def test_valid_average_macro_still_constructs() -> None:
    """Sanity: a valid ``average='macro'`` still constructs and is persisted."""
    e = ClassificationEvaluator(average="macro")
    assert e.average == "macro"


# ---------------------------------------------------------------------------
# scikit-learn optional-dependency guard (6.1). sklearn is REQUIRED in
# pyproject; the guard protects a degraded cluster env — it raises at USE
# time (not import time) so walk-discovery still registers other metrics.
# ---------------------------------------------------------------------------


def test_module_imports_and_exposes_sklearn_flag() -> None:
    """The module imports even in a torch-only env and exposes the flag."""
    import spectramr.core.metrics.classification as c

    assert isinstance(c.SKLEARN_AVAILABLE, bool)


def test_evaluator_raises_clear_error_when_sklearn_absent(monkeypatch) -> None:
    """With sklearn unavailable, evaluation raises a clear ImportError rather
    than an obscure ``NoneType is not callable`` from a stubbed metric fn."""
    import spectramr.core.metrics.classification as c

    monkeypatch.setattr(c, "SKLEARN_AVAILABLE", False)
    pred = torch.tensor([0, 1, 1, 0])
    target = torch.tensor([0, 1, 0, 0])
    with pytest.raises(ImportError, match="scikit-learn"):
        ClassificationEvaluator(num_classes=2)(pred, target)
