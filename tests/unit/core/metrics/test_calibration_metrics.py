"""Tests for calibration metrics beyond ECE (PR-8 frontier-gap audit).

Covers ``brier_score``, ``adaptive_ece`` (equal-mass bins) and
``classwise_ece``. These follow the same prediction/target convention as
the existing ``expected_calibration_error`` in
``report_step_metrics.py`` (lines 499-526): ``preds`` are predicted
probabilities/confidences (clipped to [0, 1]); ``target`` are the
binary/one-hot outcomes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Importing the module self-registers the three metrics (auto-discovery via
# pkgutil.walk_packages would also pull it in, but the explicit import keeps
# the unit test independent of package-walk ordering).
import mriforge.core.metrics.calibration_metrics  # noqa: F401
from mriforge.core.metrics.registry import MetricsRegistry, get_metric


@pytest.mark.parametrize(
    ("name", "aliases"),
    [
        ("brier_score", ["brier", "brier_loss"]),
        ("adaptive_ece", ["aece", "adaptive_calibration_error"]),
        ("classwise_ece", ["cwece", "class_wise_ece"]),
    ],
)
def test_registered_with_aliases(name: str, aliases: list[str]) -> None:
    assert MetricsRegistry.is_registered(name)
    for alias in aliases:
        assert MetricsRegistry.is_registered(alias)
        # alias resolves to the same canonical metric class
        assert get_metric(alias).__class__ is get_metric(name).__class__


@pytest.mark.parametrize("name", ["brier_score", "adaptive_ece", "classwise_ece"])
def test_lower_is_better_and_returns_float(name: str) -> None:
    metric = get_metric(name)
    assert metric.higher_is_better is False
    pred = torch.rand(64)
    target = (torch.rand(64) > 0.5).float()
    if name == "classwise_ece":
        # classwise expects per-class probability columns
        pred = torch.rand(64, 3)
        target = torch.randint(0, 3, (64,))
    out = metric(pred, target)
    assert float(out) == float(out)  # not NaN
    assert isinstance(float(out), float)


# ── brier_score ──────────────────────────────────────────────────────────────

def test_brier_zero_for_perfect_confident_correct() -> None:
    metric = get_metric("brier_score")
    # Confidence 1.0 where outcome is 1, confidence 0.0 where outcome is 0.
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    pred = target.clone()
    assert float(metric(pred, target)) == pytest.approx(0.0, abs=1e-9)


def test_brier_larger_when_wrong_or_uncertain() -> None:
    metric = get_metric("brier_score")
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    perfect = float(metric(target.clone(), target))
    uncertain = float(metric(torch.full_like(target, 0.5), target))
    wrong = float(metric(1.0 - target, target))
    assert perfect < uncertain < wrong
    assert uncertain == pytest.approx(0.25, abs=1e-9)  # (0.5)^2
    assert wrong == pytest.approx(1.0, abs=1e-9)


# ── adaptive_ece ───────────────────────────────────────────────────────────--

def test_adaptive_ece_zero_for_calibrated() -> None:
    # Construct a perfectly-calibrated set: in each confidence band the
    # empirical accuracy equals the confidence. Use deterministic blocks so
    # the equal-mass bins land cleanly.
    rng = np.random.default_rng(0)
    confs = []
    correct = []
    for c in (0.1, 0.3, 0.5, 0.7, 0.9):
        n = 2000
        confs.append(np.full(n, c))
        # exactly fraction c are correct
        block = np.zeros(n)
        block[: round(c * n)] = 1.0
        rng.shuffle(block)
        correct.append(block)
    pred = torch.tensor(np.concatenate(confs))
    target = torch.tensor(np.concatenate(correct))
    metric = get_metric("adaptive_ece", n_bins=5)
    assert float(metric(pred, target)) == pytest.approx(0.0, abs=0.02)


def test_adaptive_ece_positive_for_miscalibrated() -> None:
    # Overconfident: always claims 0.95 confidence but only 50% correct.
    pred = torch.full((1000,), 0.95)
    target = torch.tensor([1.0, 0.0] * 500)
    metric = get_metric("adaptive_ece", n_bins=10)
    assert float(metric(pred, target)) > 0.3


def test_adaptive_ece_single_unique_confidence_is_finite() -> None:
    # Degenerate: all confidences identical → quantile edges collapse to a
    # single bin. Must not crash or NaN.
    pred = torch.full((100,), 0.7)
    target = (torch.arange(100) % 2).float()
    metric = get_metric("adaptive_ece", n_bins=15)
    out = float(metric(pred, target))
    assert np.isfinite(out)
    # confidence 0.7 vs accuracy 0.5 → |0.7 - 0.5| = 0.2
    assert out == pytest.approx(0.2, abs=1e-6)


# ── classwise_ece ──────────────────────────────────────────────────────────--

def test_classwise_ece_sane_multiclass() -> None:
    rng = np.random.default_rng(1)
    n, k = 600, 3
    # Random probability simplex rows.
    logits = rng.standard_normal((n, k))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, k, size=n)
    metric = get_metric("classwise_ece", n_bins=10)
    out = float(metric(torch.tensor(probs), torch.tensor(labels)))
    assert np.isfinite(out)
    assert 0.0 <= out <= 1.0


def test_classwise_ece_zero_for_perfect_onehot() -> None:
    # One-hot predictions matching labels → every class perfectly calibrated.
    k = 4
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    probs = torch.nn.functional.one_hot(labels, k).float()
    metric = get_metric("classwise_ece", n_bins=10)
    assert float(metric(probs, labels)) == pytest.approx(0.0, abs=1e-9)


def test_classwise_ece_single_class_graceful() -> None:
    # Only one class present in targets — must not crash.
    probs = torch.rand(50, 1)
    labels = torch.zeros(50, dtype=torch.long)
    metric = get_metric("classwise_ece", n_bins=5)
    out = float(metric(probs, labels))
    assert np.isfinite(out)
