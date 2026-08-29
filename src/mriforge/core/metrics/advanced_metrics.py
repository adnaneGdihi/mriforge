"""
Phase 8.2: Advanced Metrics & Loss Analysis
Temporal, spatial, and component-level loss analysis with anomaly detection.
"""

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AnomalyType(Enum):
    """Types of anomalies in loss trajectories."""

    DIVERGENCE = "divergence"  # Loss rapidly increasing
    STAGNATION = "stagnation"  # Loss not decreasing
    SPIKE = "spike"  # Sudden jump in loss
    NAN_INF = "nan_inf"  # NaN or Inf values
    OSCILLATION = "oscillation"  # Large oscillations
    NONE = "none"  # No anomaly


@dataclass
class LossStatistics:
    """Statistics for a loss component."""

    component_name: str
    values: list[float] = field(default_factory=list)
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    trend: str = "stable"  # increasing, decreasing, stable
    latest_value: float = 0.0

    def update(self, value: float):
        """Add value and recompute statistics.

        Args:
            value: Loss value to add
        """
        self.values.append(value)
        self.latest_value = value

        if len(self.values) > 0:
            # NaN/Inf poison stats — anomaly *detection* still needs the raw
            # values around, but the running summaries must skip them or
            # ``statistics.stdev`` raises ``AttributeError`` on the NaN moment.
            finite_values = [v for v in self.values if math.isfinite(v)]
            if finite_values:
                self.mean = statistics.mean(finite_values)
                self.median = statistics.median(finite_values)
                self.min_value = min(finite_values)
                self.max_value = max(finite_values)

                if len(finite_values) > 1:
                    self.stdev = statistics.stdev(finite_values)

            # Determine trend from last 10 values
            if len(self.values) > 10:
                recent = self.values[-10:]
                recent_mean = statistics.mean(recent)
                older_mean = statistics.mean(self.values[:-10])

                if recent_mean > older_mean * 1.05:
                    self.trend = "increasing"
                elif recent_mean < older_mean * 0.95:
                    self.trend = "decreasing"
                else:
                    self.trend = "stable"

    def summary(self) -> dict[str, Any]:
        """Get statistics summary.

        Returns:
            Dictionary of statistics
        """
        return {
            "component": self.component_name,
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "stdev": round(self.stdev, 6),
            "min": round(self.min_value, 6),
            "max": round(self.max_value, 6),
            "trend": self.trend,
            "latest": round(self.latest_value, 6),
        }


@dataclass
class AnomalyReport:
    """Report of detected anomalies."""

    anomaly_type: AnomalyType
    component_name: str
    severity: float  # 0-1, higher is more severe
    description: str
    suggested_action: str
    detected_at_step: int = 0


class LossAnalyzer:
    """Analyzes loss components over time."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize loss analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.components: dict[str, LossStatistics] = {}
        self.step = 0

    def record_loss(self, loss_dict: dict[str, float]):
        """Record losses for current step.

        Args:
            loss_dict: Dict of component_name -> value
        """
        for component_name, value in loss_dict.items():
            if component_name not in self.components:
                self.components[component_name] = LossStatistics(component_name)

            self.components[component_name].update(value)

        self.step += 1

    def get_component_statistics(self, component_name: str) -> LossStatistics | None:
        """Get statistics for a component.

        Args:
            component_name: Component name

        Returns:
            LossStatistics or None
        """
        return self.components.get(component_name)

    def get_all_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all components.

        Returns:
            Dict of component_name -> statistics summary
        """
        return {name: stats.summary() for name, stats in self.components.items()}

    def detect_anomalies(self) -> list[AnomalyReport]:
        """Detect anomalies in loss trajectories.

        Returns:
            List of AnomalyReport instances
        """
        anomalies = []

        for component_name, stats in self.components.items():
            if len(stats.values) < 2:
                continue

            # Check for NaN/Inf
            if any(not np.isfinite(v) for v in stats.values):
                anomalies.append(
                    AnomalyReport(
                        anomaly_type=AnomalyType.NAN_INF,
                        component_name=component_name,
                        severity=1.0,
                        description="NaN or Inf values detected in loss",
                        suggested_action="Check numerical stability in loss computation",
                        detected_at_step=self.step,
                    )
                )
                continue

            # Check for divergence (rapid increase)
            if len(stats.values) > 5:
                recent_mean = statistics.mean(stats.values[-5:])
                older_mean = statistics.mean(stats.values[:-5])

                if recent_mean > older_mean * 2.0:
                    anomalies.append(
                        AnomalyReport(
                            anomaly_type=AnomalyType.DIVERGENCE,
                            component_name=component_name,
                            severity=0.9,
                            description=f"Loss diverging: {recent_mean:.4f} >> {older_mean:.4f}",
                            suggested_action="Reduce learning rate or check batch normalization",
                            detected_at_step=self.step,
                        )
                    )

            # Check for spike
            if len(stats.values) > 2:
                last_three = stats.values[-3:]
                if last_three[-1] > statistics.mean(last_three[:-1]) * 1.5:
                    anomalies.append(
                        AnomalyReport(
                            anomaly_type=AnomalyType.SPIKE,
                            component_name=component_name,
                            severity=0.6,
                            description=f"Loss spike detected: {last_three[-1]:.4f}",
                            suggested_action="May be a gradient explosion; consider gradient clipping",
                            detected_at_step=self.step,
                        )
                    )

            # Check for oscillation (high variance relative to mean)
            if len(stats.values) > 10 and stats.mean > 0:
                cv = stats.stdev / stats.mean  # Coefficient of variation
                if cv > 0.5:
                    anomalies.append(
                        AnomalyReport(
                            anomaly_type=AnomalyType.OSCILLATION,
                            component_name=component_name,
                            severity=0.5,
                            description=f"High oscillation: CV={cv:.2f}",
                            suggested_action="Loss is unstable; try smaller learning rate",
                            detected_at_step=self.step,
                        )
                    )

            # Check for stagnation
            if len(stats.values) > 20:
                recent_mean = statistics.mean(stats.values[-20:])
                older_mean = statistics.mean(stats.values[:-20])

                if abs(recent_mean - older_mean) / older_mean < 0.01:
                    anomalies.append(
                        AnomalyReport(
                            anomaly_type=AnomalyType.STAGNATION,
                            component_name=component_name,
                            severity=0.3,
                            description="Loss not improving",
                            suggested_action="Loss plateaued; try higher learning rate or schedule",
                            detected_at_step=self.step,
                        )
                    )

        return anomalies

    def summary(self) -> str:
        """Get analysis summary.

        Returns:
            Summary string
        """
        lines = [f"Loss Analysis Summary (Step {self.step})"]
        lines.append("=" * 70)

        for component_name, stats in self.components.items():
            lines.append(f"\n{component_name}:")
            lines.append(f"  Mean: {stats.mean:.6f}")
            lines.append(f"  Median: {stats.median:.6f}")
            lines.append(f"  Stdev: {stats.stdev:.6f}")
            lines.append(f"  Range: [{stats.min_value:.6f}, {stats.max_value:.6f}]")
            lines.append(f"  Trend: {stats.trend}")

        return "\n".join(lines)


class MetricsAggregator:
    """Aggregates metrics over training with statistics."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize metrics aggregator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self.step = 0

    def record_metrics(self, metrics_dict: dict[str, float]):
        """Record metrics for current step.

        Args:
            metrics_dict: Dict of metric_name -> value
        """
        for metric_name, value in metrics_dict.items():
            self.metrics[metric_name].append(value)

        self.step += 1

    def get_metric_statistics(self, metric_name: str) -> dict[str, float] | None:
        """Get statistics for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Dict with statistics or None
        """
        if metric_name not in self.metrics:
            return None

        values = self.metrics[metric_name]
        if not values:
            return None

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "latest": values[-1],
            "count": len(values),
        }

    def get_all_statistics(self) -> dict[str, dict[str, float]]:
        """Get statistics for all metrics.

        Returns:
            Dict of metric_name -> statistics dict
        """
        result = {}
        for metric_name in self.metrics:
            stats = self.get_metric_statistics(metric_name)
            if stats:
                result[metric_name] = stats
        return result

    def get_best_metric(self, metric_name: str, mode: str = "max") -> tuple[float, int]:
        """Get best metric value and step.

        Args:
            metric_name: Metric name
            mode: 'max' or 'min'

        Returns:
            Tuple of (value, step)
        """
        if metric_name not in self.metrics:
            return None, -1

        values = self.metrics[metric_name]
        if not values:
            return None, -1

        if mode == "max":
            best_value = max(values)
            best_step = values.index(best_value)
        else:
            best_value = min(values)
            best_step = values.index(best_value)

        return best_value, best_step

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = defaultdict(list)
        self.step = 0


class LossInteractionAnalyzer:
    """Analyzes interactions between multiple losses."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize loss interaction analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.loss_history: dict[str, list[float]] = defaultdict(list)

    def record_losses(self, loss_dict: dict[str, float]):
        """Record losses.

        Args:
            loss_dict: Dict of component_name -> value
        """
        for component_name, value in loss_dict.items():
            self.loss_history[component_name].append(value)

    def compute_correlations(self) -> dict[tuple[str, str], float]:
        """Compute correlations between loss components.

        Returns:
            Dict of (component1, component2) -> correlation coefficient
        """
        components = list(self.loss_history.keys())
        correlations = {}

        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1 :]:
                values1 = self.loss_history[comp1]
                values2 = self.loss_history[comp2]

                if len(values1) < 2 or len(values2) < 2:
                    continue

                # Compute Pearson correlation
                corr = np.corrcoef(values1, values2)[0, 1]
                correlations[(comp1, comp2)] = float(corr)

        return correlations

    def detect_redundancy(self, correlation_threshold: float = 0.9) -> list[tuple[str, str, float]]:
        """Detect redundant loss components.

        Args:
            correlation_threshold: Correlation threshold for redundancy

        Returns:
            List of (component1, component2, correlation) tuples
        """
        correlations = self.compute_correlations()
        redundant_pairs = [
            (comp1, comp2, corr)
            for (comp1, comp2), corr in correlations.items()
            if abs(corr) > correlation_threshold
        ]
        return redundant_pairs

    def detect_synergy(self, correlation_threshold: float = -0.5) -> list[tuple[str, str, float]]:
        """Detect synergistic loss components.

        Args:
            correlation_threshold: Correlation threshold for synergy

        Returns:
            List of (component1, component2, correlation) tuples
        """
        correlations = self.compute_correlations()
        synergistic_pairs = [
            (comp1, comp2, corr)
            for (comp1, comp2), corr in correlations.items()
            if corr < correlation_threshold
        ]
        return synergistic_pairs

    def compute_loss_balance(self) -> dict[str, float]:
        """Compute balance of losses (relative contribution).

        Returns:
            Dict of component_name -> balance (0-1)
        """
        balance = {}

        if not self.loss_history:
            return balance

        # Use latest values
        latest_values = {name: values[-1] for name, values in self.loss_history.items()}

        total = sum(abs(v) for v in latest_values.values())
        if total > 0:
            balance = {name: abs(v) / total for name, v in latest_values.items()}

        return balance

    def summary(self) -> str:
        """Get interaction summary.

        Returns:
            Summary string
        """
        lines = ["Loss Interaction Analysis"]
        lines.append("=" * 70)

        # Correlations
        correlations = self.compute_correlations()
        if correlations:
            lines.append("\nCorrelations:")
            for (comp1, comp2), corr in sorted(
                correlations.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                lines.append(f"  {comp1} <-> {comp2}: {corr:.4f}")

        # Redundancy
        redundant = self.detect_redundancy()
        if redundant:
            lines.append("\nRedundant Components:")
            for comp1, comp2, corr in redundant:
                lines.append(f"  {comp1} and {comp2} (corr={corr:.4f})")

        # Synergy
        synergistic = self.detect_synergy()
        if synergistic:
            lines.append("\nSynergistic Components:")
            for comp1, comp2, corr in synergistic:
                lines.append(f"  {comp1} <-> {comp2} (corr={corr:.4f})")

        # Balance
        balance = self.compute_loss_balance()
        if balance:
            lines.append("\nLoss Balance:")
            for name, bal in sorted(balance.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {name}: {bal:.1%}")

        return "\n".join(lines)
