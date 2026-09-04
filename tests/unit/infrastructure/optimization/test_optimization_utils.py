"""Unit tests for optimization utility dataclasses.

Targets ``spectramr.infrastructure.training.optimization_utils``:
- ``GradientBuffer``: store/retrieve/evict/hit-rate
- ``AsyncMetricsReporter``: report_async / flush / aggregate
- ``OptimizationMetrics``: record_timing / get_stats / reset
- ``PerformanceContextManager``: context-manager timing

CPU-only; no GPU dependency.
"""

from __future__ import annotations

import time

import pytest
import torch

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# GradientBuffer
# ---------------------------------------------------------------------------


def test_canary_gradient_buffer_store_retrieve() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer(capacity=10)
    t = torch.randn(4, 4)
    buf.store("key1", t)
    result = buf.retrieve("key1")
    assert result is t


@pytest.mark.parametrize("n_items,capacity", [(5, 10), (10, 10), (15, 10)])
def test_gradient_buffer_capacity_eviction(n_items: int, capacity: int) -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer(capacity=capacity)
    tensors = []
    for i in range(n_items):
        t = torch.ones(2)
        buf.store(f"key_{i}", t)
        tensors.append(t)  # keep strong refs so weak refs stay alive

    assert buf.stats()["current_size"] <= capacity


def test_gradient_buffer_miss_returns_none() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer()
    assert buf.retrieve("__nonexistent__") is None


def test_gradient_buffer_hit_rate_all_hits() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer()
    t = torch.ones(2)
    buf.store("k", t)
    buf.retrieve("k")
    buf.retrieve("k")
    assert buf.hit_rate() == pytest.approx(1.0)


def test_gradient_buffer_hit_rate_all_misses() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer()
    buf.retrieve("__x__")
    buf.retrieve("__y__")
    assert buf.hit_rate() == pytest.approx(0.0)


def test_gradient_buffer_clear_resets_stats() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer()
    t = torch.ones(2)
    buf.store("k", t)
    buf.retrieve("k")
    buf.clear()
    assert buf.stats()["hits"] == 0
    assert buf.stats()["stores"] == 0
    assert buf.stats()["current_size"] == 0


# ---------------------------------------------------------------------------
# AsyncMetricsReporter
# ---------------------------------------------------------------------------


def test_async_reporter_report_and_flush() -> None:
    from spectramr.infrastructure.training.optimization_utils import AsyncMetricsReporter

    collected: list = []
    reporter = AsyncMetricsReporter(batch_size=1)
    reporter.set_callback(lambda m: collected.append(m))
    reporter.report_async({"loss": 0.5})
    reporter.flush()
    # At least one aggregated entry recorded.
    assert len(collected) >= 1


def test_async_reporter_aggregate_averages() -> None:
    from spectramr.infrastructure.training.optimization_utils import AsyncMetricsReporter

    reporter = AsyncMetricsReporter(batch_size=3)
    batch = [{"a": 1.0}, {"a": 3.0}, {"a": 5.0}]
    agg = reporter._aggregate_metrics(batch)
    assert agg["a"] == pytest.approx(3.0)  # mean of 1, 3, 5


def test_async_reporter_aggregate_empty_returns_empty() -> None:
    from spectramr.infrastructure.training.optimization_utils import AsyncMetricsReporter

    reporter = AsyncMetricsReporter()
    assert reporter._aggregate_metrics([]) == {}


# ---------------------------------------------------------------------------
# OptimizationMetrics
# ---------------------------------------------------------------------------


def test_optimization_metrics_record_and_stats() -> None:
    from spectramr.infrastructure.training.optimization_utils import OptimizationMetrics

    metrics = OptimizationMetrics(name="test")
    for v in [10.0, 20.0, 30.0]:
        metrics.record_timing("generator_forward", v)
    stats = metrics.get_stats("generator_forward")
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(20.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(30.0)


def test_optimization_metrics_empty_component_returns_empty_dict() -> None:
    from spectramr.infrastructure.training.optimization_utils import OptimizationMetrics

    metrics = OptimizationMetrics()
    assert metrics.get_stats("__nonexistent__") == {}


def test_optimization_metrics_reset_clears_timings() -> None:
    from spectramr.infrastructure.training.optimization_utils import OptimizationMetrics

    metrics = OptimizationMetrics()
    metrics.record_timing("backward_pass", 5.0)
    metrics.reset()
    assert metrics.get_stats("backward_pass") == {}


# ---------------------------------------------------------------------------
# PerformanceContextManager
# ---------------------------------------------------------------------------


def test_performance_context_manager_records_timing() -> None:
    from spectramr.infrastructure.training.optimization_utils import (
        OptimizationMetrics,
        PerformanceContextManager,
    )

    metrics = OptimizationMetrics()
    with PerformanceContextManager("metrics_logging", metrics):
        pass  # near-instant work

    stats = metrics.get_stats("metrics_logging")
    assert stats["count"] == 1
    # Duration must be non-negative
    assert stats["min"] >= 0.0


# ---------------------------------------------------------------------------
# Edge: GradientBuffer empty hit_rate
# ---------------------------------------------------------------------------


def test_gradient_buffer_edge_zero_queries_hit_rate() -> None:
    from spectramr.infrastructure.training.optimization_utils import GradientBuffer

    buf = GradientBuffer()
    # No stores, no retrieves → hit_rate defaults to 0.0
    assert buf.hit_rate() == pytest.approx(0.0)
