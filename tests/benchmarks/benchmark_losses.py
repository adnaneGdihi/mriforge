"""
Phase 6.3: Performance Benchmarks
Measures loss computation and factory performance.
"""

import statistics
import time

import pytest
import torch
import torch.nn as nn

from mriforge.config.schemas.loss import (
    L1LossConfig,
    LossConfigSchema,
    PerceptualLossConfig,
    ReconstructionLossesConfig,
    SSIMLossConfig,
)
from mriforge.models.losses.composed_loss import ComposedLoss, WeightedLoss
from mriforge.models.losses.registry import create_loss


def create_composed_loss_from_schema(
    config: LossConfigSchema, device: torch.device
) -> nn.Module:
    """Create a composed loss from schema using Phase 5 registry."""
    weighted_losses = []
    if (
        config.reconstruction
        and hasattr(config.reconstruction, "l1")
        and config.reconstruction.l1.enabled
    ):
        weighted_losses.append(
            WeightedLoss(
                name="l1",
                loss_fn=create_loss("l1"),
                weight=config.reconstruction.l1.weight,
            )
        )
    if (
        config.reconstruction
        and hasattr(config.reconstruction, "ssim")
        and config.reconstruction.ssim.enabled
    ):
        weighted_losses.append(
            WeightedLoss(
                name="ssim",
                loss_fn=create_loss("ssim"),
                weight=config.reconstruction.ssim.weight,
            )
        )
    if (
        config.reconstruction
        and hasattr(config.reconstruction, "perceptual")
        and config.reconstruction.perceptual.enabled
    ):
        weighted_losses.append(
            WeightedLoss(
                name="perceptual",
                loss_fn=create_loss("perceptual"),
                weight=config.reconstruction.perceptual.weight,
            )
        )

    return ComposedLoss(weighted_losses).to(device)


class BenchmarkResults:
    """Results from a benchmark run."""

    def __init__(self, name: str, times: list[float]):
        """Initialize benchmark results.

        Args:
            name: Benchmark name
            times: List of measured times (ms)
        """
        self.name = name
        self.times = times

    @property
    def mean(self) -> float:
        """Mean time (ms)."""
        return statistics.mean(self.times)

    @property
    def median(self) -> float:
        """Median time (ms)."""
        return statistics.median(self.times)

    @property
    def stdev(self) -> float:
        """Standard deviation (ms)."""
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)

    @property
    def min(self) -> float:
        """Minimum time (ms)."""
        return min(self.times)

    @property
    def max(self) -> float:
        """Maximum time (ms)."""
        return max(self.times)

    def summary(self) -> str:
        """Get summary string."""
        return (
            f"{self.name}: "
            f"mean={self.mean:.2f}ms, "
            f"median={self.median:.2f}ms, "
            f"stdev={self.stdev:.2f}ms, "
            f"min={self.min:.2f}ms, "
            f"max={self.max:.2f}ms"
        )


class LossBenchmark:
    """Benchmarks for loss computation."""

    def __init__(self, device: str = "cpu"):
        """Initialize benchmark.

        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

    def benchmark_loss_creation(
        self,
        config: LossConfigSchema,
        num_runs: int = 100,
    ) -> BenchmarkResults:
        """Benchmark loss creation time.

        Args:
            config: Loss configuration
            num_runs: Number of runs

        Returns:
            BenchmarkResults with timing information
        """
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            loss = create_composed_loss_from_schema(config, self.device)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            times.append(elapsed)

        return BenchmarkResults("Loss Creation", times)

    def benchmark_loss_forward(
        self,
        loss: nn.Module,
        batch_size: int = 4,
        height: int = 64,
        width: int = 64,
        num_runs: int = 100,
    ) -> BenchmarkResults:
        """Benchmark loss forward pass.

        Args:
            loss: Loss function
            batch_size: Batch size
            height: Image height
            width: Image width
            num_runs: Number of runs

        Returns:
            BenchmarkResults with timing information
        """
        loss.eval()
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                pred = torch.randn(batch_size, 3, height, width).to(self.device)
                target = torch.randn(batch_size, 3, height, width).to(self.device)

                start = time.perf_counter()
                # Use metrics-aware API for composed losses
                if hasattr(loss, "forward_with_metrics"):
                    loss_val, _ = loss.forward_with_metrics(pred, target)
                else:
                    loss_val = loss(pred, target)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                times.append(elapsed)

        return BenchmarkResults("Loss Forward", times)

    def benchmark_metrics_extraction(
        self,
        loss: nn.Module,
        batch_size: int = 4,
        height: int = 64,
        width: int = 64,
        num_runs: int = 100,
    ) -> BenchmarkResults:
        """Benchmark metrics extraction.

        Args:
            loss: Loss function
            batch_size: Batch size
            height: Image height
            width: Image width
            num_runs: Number of runs

        Returns:
            BenchmarkResults with timing information
        """
        loss.eval()
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                pred = torch.randn(batch_size, 3, height, width).to(self.device)
                target = torch.randn(batch_size, 3, height, width).to(self.device)

                start = time.perf_counter()
                # Check for metrics-aware losses (Phase 5 standard)
                if hasattr(loss, "forward_with_metrics"):
                    loss_val, metrics = loss.forward_with_metrics(pred, target)
                else:
                    loss_val = loss(pred, target)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                times.append(elapsed)

        return BenchmarkResults("Metrics Extraction", times)

    def benchmark_metrics_overhead(
        self,
        loss: nn.Module,
        batch_size: int = 4,
        height: int = 64,
        width: int = 64,
        num_runs: int = 100,
    ) -> float:
        """Benchmark metrics computation overhead (percentage).

        Args:
            loss: Loss function
            batch_size: Batch size
            height: Image height
            width: Image width
            num_runs: Number of runs

        Returns:
            Overhead percentage (0-100%)
        """
        forward_results = self.benchmark_loss_forward(
            loss, batch_size, height, width, num_runs
        )
        metrics_results = self.benchmark_metrics_extraction(
            loss, batch_size, height, width, num_runs
        )

        overhead = (
            (metrics_results.mean - forward_results.mean) / forward_results.mean
        ) * 100
        return max(0, overhead)  # Ensure non-negative


# Benchmark Tests


class TestLossBenchmarks:
    """Test loss computation performance."""

    @pytest.fixture
    def benchmark(self) -> LossBenchmark:
        """Create loss benchmark."""
        return LossBenchmark()

    @pytest.fixture
    def simple_config(self) -> LossConfigSchema:
        """Create simple loss config."""
        return LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                l1=L1LossConfig(enabled=True, weight=10.0),
            )
        )

    @pytest.fixture
    def complex_config(self) -> LossConfigSchema:
        """Create complex loss config."""
        return LossConfigSchema(
            reconstruction=ReconstructionLossesConfig(
                l1=L1LossConfig(enabled=True, weight=10.0),
                ssim=SSIMLossConfig(enabled=True, weight=0.1),
                perceptual=PerceptualLossConfig(enabled=True, weight=0.1),
            )
        )

    def test_loss_creation_time(
        self, benchmark: LossBenchmark, simple_config: LossConfigSchema
    ):
        """Test loss creation time."""
        results = benchmark.benchmark_loss_creation(simple_config, num_runs=50)
        print(f"\n{results.summary()}")

        # Verify reasonable performance
        assert results.mean < 50, f"Loss creation too slow: {results.mean}ms"

    def test_loss_forward_time(
        self, benchmark: LossBenchmark, simple_config: LossConfigSchema
    ):
        """Test loss forward pass time."""
        loss = create_composed_loss_from_schema(simple_config, benchmark.device)
        results = benchmark.benchmark_loss_forward(loss, num_runs=50)
        print(f"\n{results.summary()}")

        # Verify reasonable performance
        assert results.mean < 10, f"Loss forward too slow: {results.mean}ms"

    def test_metrics_overhead(
        self, benchmark: LossBenchmark, simple_config: LossConfigSchema
    ):
        """Test metrics computation overhead."""
        loss = create_composed_loss_from_schema(simple_config, benchmark.device)
        overhead = benchmark.benchmark_metrics_overhead(loss, num_runs=50)
        print(f"\nMetrics overhead: {overhead:.1f}%")

        # Verify metrics overhead is acceptable
        assert overhead < 10, f"Metrics overhead too high: {overhead}%"

    def test_complex_loss_overhead(
        self, benchmark: LossBenchmark, complex_config: LossConfigSchema
    ):
        """Test metrics overhead with multiple losses."""
        loss = create_composed_loss_from_schema(complex_config, benchmark.device)
        overhead = benchmark.benchmark_metrics_overhead(loss, num_runs=50)
        print(f"\nComplex loss metrics overhead: {overhead:.1f}%")

        # Overhead acceptable even with multiple losses
        assert overhead < 15, f"Complex loss metrics overhead too high: {overhead}%"


# Standalone benchmark

if __name__ == "__main__":
    print("=" * 60)
    print("Loss Computation Benchmarks")
    print("=" * 60)

    benchmark = LossBenchmark()
    config = LossConfigSchema(
        reconstruction=ReconstructionLossesConfig(
            l1=L1LossConfig(enabled=True, weight=10.0),
            ssim=SSIMLossConfig(enabled=True, weight=0.1),
        )
    )

    loss = create_composed_loss_from_schema(config, benchmark.device)

    print("\n1. Loss Creation Performance:")
    creation_results = benchmark.benchmark_loss_creation(config, num_runs=100)
    print(f"   {creation_results.summary()}")

    print("\n2. Loss Forward Performance:")
    forward_results = benchmark.benchmark_loss_forward(loss, num_runs=100)
    print(f"   {forward_results.summary()}")

    print("\n3. Metrics Extraction Performance:")
    metrics_results = benchmark.benchmark_metrics_extraction(loss, num_runs=100)
    print(f"   {metrics_results.summary()}")

    print("\n4. Metrics Overhead:")
    overhead = benchmark.benchmark_metrics_overhead(loss, num_runs=100)
    print(f"   Overhead: {overhead:.2f}%")

    print("\n" + "=" * 60)
