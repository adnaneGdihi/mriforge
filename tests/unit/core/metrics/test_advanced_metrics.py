"""Tests for ``LossStatistics`` and ``LossAnalyzer``.

Targets ``spectramr.core.metrics.advanced_metrics``. Provides streaming
statistics + anomaly detection (divergence, NaN/Inf, stagnation,
spike, oscillation) over a loss trajectory.

The full ``MetricsAggregator`` and ``LossInteractionAnalyzer`` are
covered separately — focusing here on the building blocks that the
training loop actually calls.
"""

from __future__ import annotations

import math

from spectramr.core.metrics.advanced_metrics import (
    AnomalyReport,
    AnomalyType,
    LossAnalyzer,
    LossStatistics,
)


# ---------------------------------------------------------------------------
# AnomalyType enum
# ---------------------------------------------------------------------------


def test_anomaly_type_enum_covers_documented_categories() -> None:
    """All five anomaly categories + NONE are present."""
    expected = {"divergence", "stagnation", "spike", "nan_inf", "oscillation", "none"}
    actual = {member.value for member in AnomalyType}
    assert expected <= actual


# ---------------------------------------------------------------------------
# LossStatistics — single-component lifecycle
# ---------------------------------------------------------------------------


def test_loss_stats_initial_state_empty() -> None:
    """Fresh ``LossStatistics`` has empty values + zero scalars."""
    s = LossStatistics(component_name="recon_l1")
    assert s.values == []
    assert s.mean == 0.0
    assert s.trend == "stable"


def test_loss_stats_update_records_value_and_recomputes_mean() -> None:
    """``update(v)`` appends value and refreshes mean / median / latest."""
    s = LossStatistics(component_name="recon_l1")
    s.update(1.0)
    s.update(3.0)
    assert s.mean == 2.0
    assert s.median == 2.0
    assert s.latest_value == 3.0
    assert s.min_value == 1.0
    assert s.max_value == 3.0


def test_loss_stats_stdev_requires_two_samples() -> None:
    """``stdev`` is computed from the second sample onward."""
    s = LossStatistics(component_name="x")
    s.update(2.0)
    assert s.stdev == 0.0  # no stdev with one sample
    s.update(4.0)
    assert s.stdev > 0


def test_loss_stats_trend_increasing() -> None:
    """When recent values exceed older ones by >5%, trend is ``increasing``."""
    s = LossStatistics(component_name="x")
    # Older 5 values around 1.0
    for _ in range(5):
        s.update(1.0)
    # 6 more older values around 1.0 (need >10 total to evaluate trend)
    for _ in range(6):
        s.update(1.0)
    # Now overwrite recent block (last 10) with values > 1.05
    for _ in range(10):
        s.update(1.5)
    assert s.trend == "increasing"


def test_loss_stats_trend_decreasing() -> None:
    """When recent values are below older ones by >5%, trend is ``decreasing``."""
    s = LossStatistics(component_name="x")
    for _ in range(11):
        s.update(2.0)
    for _ in range(10):
        s.update(1.0)
    assert s.trend == "decreasing"


def test_loss_stats_summary_keys() -> None:
    """Summary dict contains the 8 documented keys."""
    s = LossStatistics(component_name="x")
    s.update(1.0)
    summary = s.summary()
    expected = {"component", "mean", "median", "stdev", "min", "max", "trend", "latest"}
    assert summary.keys() == expected


# ---------------------------------------------------------------------------
# LossAnalyzer — record + retrieve
# ---------------------------------------------------------------------------


def test_record_loss_creates_component_on_first_call() -> None:
    """First ``record_loss`` call instantiates the per-component stats."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"recon_l1": 0.5, "perceptual": 0.3})
    assert "recon_l1" in analyzer.components
    assert "perceptual" in analyzer.components
    assert analyzer.step == 1


def test_record_loss_increments_step() -> None:
    """Each ``record_loss`` call increments the global step counter."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"x": 1.0})
    analyzer.record_loss({"x": 1.0})
    analyzer.record_loss({"x": 1.0})
    assert analyzer.step == 3


def test_get_component_statistics_returns_existing() -> None:
    """``get_component_statistics`` returns the stats object for a known component."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"recon": 0.5})
    stats = analyzer.get_component_statistics("recon")
    assert stats is not None
    assert stats.component_name == "recon"
    assert stats.mean == 0.5


def test_get_component_statistics_returns_none_for_unknown() -> None:
    """Unknown component → None (no exception)."""
    analyzer = LossAnalyzer()
    assert analyzer.get_component_statistics("ghost") is None


def test_get_all_statistics_returns_summaries() -> None:
    """``get_all_statistics`` returns one summary dict per recorded component."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"a": 0.1, "b": 0.2})
    all_stats = analyzer.get_all_statistics()
    assert {"a", "b"} == all_stats.keys()
    assert all_stats["a"]["latest"] == 0.1


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def test_detect_anomalies_returns_empty_for_short_trajectory() -> None:
    """A single-value trajectory cannot trigger anomalies."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"x": 0.5})
    assert analyzer.detect_anomalies() == []


def test_detect_anomalies_flags_nan() -> None:
    """NaN values produce a ``NAN_INF`` anomaly with severity 1.0."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"x": 0.5})
    analyzer.record_loss({"x": float("nan")})
    reports = analyzer.detect_anomalies()
    nan_reports = [r for r in reports if r.anomaly_type == AnomalyType.NAN_INF]
    assert len(nan_reports) == 1
    assert nan_reports[0].severity == 1.0
    assert nan_reports[0].component_name == "x"


def test_detect_anomalies_flags_inf() -> None:
    """Inf values also trigger ``NAN_INF``."""
    analyzer = LossAnalyzer()
    analyzer.record_loss({"x": 0.5})
    analyzer.record_loss({"x": math.inf})
    reports = analyzer.detect_anomalies()
    assert any(r.anomaly_type == AnomalyType.NAN_INF for r in reports)


def test_detect_anomalies_flags_divergence() -> None:
    """Loss doubling in the last 5 steps triggers ``DIVERGENCE``."""
    analyzer = LossAnalyzer()
    # Older steps: stable around 0.5
    for _ in range(10):
        analyzer.record_loss({"x": 0.5})
    # Recent 5 steps: drastically larger
    for _ in range(5):
        analyzer.record_loss({"x": 10.0})
    reports = analyzer.detect_anomalies()
    assert any(r.anomaly_type == AnomalyType.DIVERGENCE for r in reports)


# ---------------------------------------------------------------------------
# AnomalyReport dataclass
# ---------------------------------------------------------------------------


def test_anomaly_report_fields() -> None:
    """``AnomalyReport`` has the documented fields."""
    report = AnomalyReport(
        anomaly_type=AnomalyType.NONE,
        component_name="x",
        severity=0.0,
        description="ok",
        suggested_action="",
        detected_at_step=0,
    )
    assert report.anomaly_type == AnomalyType.NONE
    assert report.component_name == "x"
    assert report.severity == 0.0
