"""Pool hygiene — clone and broken-metric screening (critique §8 repair)."""

from __future__ import annotations

import torch

from spectramr.core.metrics.meta_evaluation.pool_hygiene import screen_pool
from spectramr.core.metrics.meta_evaluation.types import (
    DegradationSample,
    MetricEvaluationDataset,
)

N = 12


def _dataset(metric_values: dict[str, list[float]]) -> MetricEvaluationDataset:
    # Residual energy increases monotonically with sample index (the proxy).
    samples = []
    for i in range(N):
        clean = torch.ones(4)
        degraded = clean - (i + 1) * 0.1
        samples.append(
            DegradationSample(
                clean=clean,
                degraded=degraded,
                degradation_family="gaussian_noise",
                severity={"sigma": (i + 1) * 0.1},
                content_id=f"c{i % 3}",
                seed=i,
            )
        )
    return MetricEvaluationDataset(samples=samples, metric_values=metric_values)


def _values() -> dict[str, list[float]]:
    return {
        "good": [float(i) for i in range(N)],  # tracks the degradation proxy
        "clone": [3.0 * i + 7.0 for i in range(N)],  # monotone image of good
        "broken_const": [2.0] * N,  # zero variance
        "broken_tent": [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0],  # ~0 corr with proxy
    }


def test_broken_metrics_detected() -> None:
    s = screen_pool(_dataset(_values()))
    assert "broken_const" in s.broken
    assert "broken_tent" in s.broken
    assert "good" not in s.broken
    assert "clone" not in s.broken


def test_clones_clustered_one_canonical() -> None:
    s = screen_pool(_dataset(_values()))
    # good and clone collapse into a single cluster, one canonical voter.
    assert any({"good", "clone"} <= set(c) for c in s.clone_clusters)
    assert ("good" in s.canonical) ^ ("clone" in s.canonical)  # exactly one votes
    assert len(s.redundant) == 1
    # Voting pool excludes both broken metrics and the redundant clone.
    assert "broken_const" not in s.voting_metrics
    assert "broken_tent" not in s.voting_metrics
    assert len(s.voting_metrics) == 1


def test_clone_threshold_above_one_keeps_both() -> None:
    # No pair can reach |rho| >= 1.01, so good and clone both vote.
    s = screen_pool(_dataset(_values()), clone_threshold=1.01)
    assert s.clone_clusters == []
    assert set(s.voting_metrics) == {"good", "clone"}


def test_signal_threshold_zero_keeps_noise() -> None:
    # With signal_threshold 0, only the zero-variance metric is broken.
    s = screen_pool(_dataset(_values()), signal_threshold=0.0)
    assert "broken_const" in s.broken
    assert "broken_tent" not in s.broken


def test_signal_reported_per_metric() -> None:
    s = screen_pool(_dataset(_values()))
    # 'good' is perfectly monotone in the proxy -> |Spearman| ~ 1.
    assert s.signal["good"] > 0.99
    assert s.signal["broken_tent"] < 0.1
