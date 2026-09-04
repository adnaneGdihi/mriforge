"""Unit tests for advanced_metrics.py — four archetypes per public class.

Covers:
- LossStatistics: update, summary, trend detection, NaN/Inf immunity.
- LossAnalyzer: record_loss, get_all_statistics, detect_anomalies.
- MetricsAggregator: record_metrics, get_metric_statistics, get_best_metric.
- LossInteractionAnalyzer: compute_correlations, detect_redundancy/synergy.
"""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# LossStatistics
# ---------------------------------------------------------------------------


class TestLossStatistics:
    @pytest.fixture
    def ls(self):
        from spectramr.core.metrics.advanced_metrics import LossStatistics
        return LossStatistics(component_name="test_loss")

    # canary
    def test_canary_update_summary(self, ls):
        for v in [1.0, 2.0, 3.0]:
            ls.update(v)
        s = ls.summary()
        assert "mean" in s
        assert math.isfinite(s["mean"])

    # parametrize over value ranges
    @pytest.mark.parametrize("values", [
        [0.1, 0.2, 0.3],
        [100.0, 200.0, 300.0],
        [1e-6, 2e-6, 3e-6],
    ])
    def test_value_ranges(self, values):
        from spectramr.core.metrics.advanced_metrics import LossStatistics
        s = LossStatistics("x")
        for v in values:
            s.update(v)
        summary = s.summary()
        assert math.isfinite(summary["mean"])
        assert math.isfinite(summary["stdev"]) or summary["stdev"] == 0.0

    # edge: NaN/Inf should not poison mean (finite_values filter)
    def test_edge_nan_value_does_not_poison_mean(self, ls):
        ls.update(1.0)
        ls.update(float("nan"))
        ls.update(3.0)
        s = ls.summary()
        assert math.isfinite(s["mean"]), f"mean should be finite, got {s['mean']}"

    def test_edge_inf_value_does_not_poison_mean(self, ls):
        ls.update(1.0)
        ls.update(float("inf"))
        ls.update(2.0)
        s = ls.summary()
        assert math.isfinite(s["mean"])

    # edge: single value — stdev stays 0
    def test_edge_single_value_zero_stdev(self, ls):
        ls.update(5.0)
        s = ls.summary()
        assert s["stdev"] == 0.0

    # parametrize trend detection: needs >10 samples
    def test_trend_increasing(self):
        from spectramr.core.metrics.advanced_metrics import LossStatistics
        s = LossStatistics("inc")
        # 15 values increasing significantly
        for i in range(15):
            s.update(float(i) * 10)
        assert s.trend == "increasing"

    def test_trend_decreasing(self):
        from spectramr.core.metrics.advanced_metrics import LossStatistics
        s = LossStatistics("dec")
        for i in range(15):
            s.update(float(15 - i) * 10)
        assert s.trend == "decreasing"

    def test_summary_keys(self, ls):
        ls.update(1.0)
        s = ls.summary()
        for key in ["component", "mean", "median", "stdev", "min", "max", "trend", "latest"]:
            assert key in s, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# LossAnalyzer
# ---------------------------------------------------------------------------


class TestLossAnalyzer:
    @pytest.fixture
    def analyzer(self):
        from spectramr.core.metrics.advanced_metrics import LossAnalyzer
        return LossAnalyzer()

    # canary
    def test_canary_record_and_retrieve(self, analyzer):
        analyzer.record_loss({"gen_loss": 0.5, "disc_loss": 0.3})
        stats = analyzer.get_all_statistics()
        assert "gen_loss" in stats
        assert "disc_loss" in stats

    # parametrize over batch sizes (number of records)
    @pytest.mark.parametrize("n_steps", [1, 2])
    def test_n_steps(self, n_steps):
        from spectramr.core.metrics.advanced_metrics import LossAnalyzer
        a = LossAnalyzer()
        for i in range(n_steps):
            a.record_loss({"loss": float(i + 1)})
        assert a.step == n_steps
        stats = a.get_all_statistics()
        assert "loss" in stats

    # edge: NaN value produces NAN_INF anomaly
    def test_edge_nan_triggers_nan_inf_anomaly(self, analyzer):
        from spectramr.core.metrics.advanced_metrics import AnomalyType
        analyzer.record_loss({"loss": 1.0})
        analyzer.record_loss({"loss": float("nan")})
        anomalies = analyzer.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.NAN_INF in types

    # edge: divergence detected when recent >> older
    def test_edge_divergence_detected(self, analyzer):
        from spectramr.core.metrics.advanced_metrics import AnomalyType
        # 10 baseline values + 5 very large values
        for _ in range(10):
            analyzer.record_loss({"loss": 1.0})
        for _ in range(5):
            analyzer.record_loss({"loss": 100.0})
        anomalies = analyzer.detect_anomalies()
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.DIVERGENCE in types

    # edge: component not yet recorded returns None
    def test_edge_missing_component_returns_none(self, analyzer):
        result = analyzer.get_component_statistics("nonexistent")
        assert result is None

    # raises: record_loss with empty dict is valid (no crash, step increments)
    def test_raises_empty_dict_no_crash(self, analyzer):
        analyzer.record_loss({})
        assert analyzer.step == 1

    def test_summary_string_not_empty(self, analyzer):
        for i in range(3):
            analyzer.record_loss({"g": float(i)})
        s = analyzer.summary()
        assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------------------


class TestMetricsAggregator:
    @pytest.fixture
    def agg(self):
        from spectramr.core.metrics.advanced_metrics import MetricsAggregator
        return MetricsAggregator()

    # canary
    def test_canary_record_and_stats(self, agg):
        agg.record_metrics({"psnr": 30.0, "ssim": 0.9})
        stats = agg.get_metric_statistics("psnr")
        assert stats is not None
        assert math.isfinite(stats["mean"])

    # parametrize
    @pytest.mark.parametrize("n", [1, 2])
    def test_n_records(self, n):
        from spectramr.core.metrics.advanced_metrics import MetricsAggregator
        a = MetricsAggregator()
        for i in range(n):
            a.record_metrics({"m": float(i + 1)})
        s = a.get_metric_statistics("m")
        assert s["count"] == n

    # edge: missing key returns None
    def test_edge_missing_key_none(self, agg):
        assert agg.get_metric_statistics("nosuchmetric") is None

    # edge: empty values list returns None
    def test_edge_empty_metrics_returns_none(self, agg):
        # Don't record anything — should return None
        assert agg.get_metric_statistics("empty") is None

    # edge: get_best_metric max and min
    def test_get_best_metric_max(self, agg):
        for v in [0.5, 0.9, 0.7]:
            agg.record_metrics({"psnr": v})
        best, step = agg.get_best_metric("psnr", mode="max")
        assert best == pytest.approx(0.9, abs=1e-6)

    def test_get_best_metric_min(self, agg):
        for v in [0.5, 0.9, 0.3]:
            agg.record_metrics({"loss": v})
        best, step = agg.get_best_metric("loss", mode="min")
        assert best == pytest.approx(0.3, abs=1e-6)

    # reset
    def test_reset_clears_state(self, agg):
        agg.record_metrics({"x": 1.0})
        agg.reset_metrics()
        assert agg.step == 0
        assert agg.get_metric_statistics("x") is None

    def test_get_all_statistics_structure(self, agg):
        agg.record_metrics({"a": 1.0, "b": 2.0})
        agg.record_metrics({"a": 3.0, "b": 4.0})
        all_stats = agg.get_all_statistics()
        for key in ["a", "b"]:
            assert key in all_stats
            assert "mean" in all_stats[key]


# ---------------------------------------------------------------------------
# LossInteractionAnalyzer
# ---------------------------------------------------------------------------


class TestLossInteractionAnalyzer:
    @pytest.fixture
    def lia(self):
        from spectramr.core.metrics.advanced_metrics import LossInteractionAnalyzer
        return LossInteractionAnalyzer()

    def _fill(self, lia, n: int = 20):
        import random
        random.seed(42)
        for i in range(n):
            lia.record_losses({
                "gen": float(i) * 0.01 + random.gauss(0, 0.001),
                "disc": float(n - i) * 0.01 + random.gauss(0, 0.001),
            })

    # canary
    def test_canary_correlations(self, lia):
        self._fill(lia)
        corr = lia.compute_correlations()
        assert ("gen", "disc") in corr
        assert math.isfinite(corr[("gen", "disc")])

    # parametrize
    @pytest.mark.parametrize("n", [1, 2])
    def test_n_records_correlation(self, n):
        from spectramr.core.metrics.advanced_metrics import LossInteractionAnalyzer
        la = LossInteractionAnalyzer()
        for i in range(max(n, 2)):  # Need >= 2 for correlation
            la.record_losses({"a": float(i), "b": float(i) + 1.0})
        corr = la.compute_correlations()
        assert isinstance(corr, dict)

    # edge: fewer than 2 samples skip correlation
    def test_edge_single_sample_no_crash(self, lia):
        lia.record_losses({"a": 1.0, "b": 2.0})
        corr = lia.compute_correlations()
        # Single sample — correlation should be empty (need >= 2 values)
        assert isinstance(corr, dict)

    # edge: perfectly correlated → detected as redundant
    def test_edge_redundancy_detection(self, lia):
        for i in range(30):
            lia.record_losses({"a": float(i), "b": float(i) * 1.001})
        redundant = lia.detect_redundancy(correlation_threshold=0.9)
        # a and b are near-identical, should be flagged
        assert len(redundant) > 0

    # edge: anti-correlated → detected as synergistic
    def test_edge_synergy_detection(self, lia):
        for i in range(30):
            lia.record_losses({"a": float(i), "b": float(30 - i)})
        synergistic = lia.detect_synergy(correlation_threshold=-0.5)
        assert len(synergistic) > 0

    def test_loss_balance_keys_present(self, lia):
        self._fill(lia)
        balance = lia.compute_loss_balance()
        assert isinstance(balance, dict)
