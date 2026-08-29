"""Performance benchmarks for DataLoader throughput.

Measures:
- Samples per second throughput
- Batch loading time
- Pin memory impact
- Worker process efficiency
"""

import time
from pathlib import Path

import pytest
import torch

from mriforge.config.settings import TrainingSettings
from mriforge.infrastructure.builders.context import BuilderContext
from mriforge.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)


def _build_loaders(config):
    """Build (train, val, None) through the canonical director.

    The old call this replaces was ``ConsolidatedDatasetFactory().
    create_data_loaders(config)`` -- and that method never existed on the class,
    which defined only ``create`` and ``create_datamodule``. This benchmark
    therefore raised ``AttributeError`` and had not run since it was written.

    The third element was always a fiction: the director builds train and val,
    and nothing here reads the third. It is returned as ``None`` so the call
    sites keep their shape rather than being rewritten blind.
    """
    train, val = DataPipelineDirector(BuilderContext(config=config)).build_dataloaders()
    return train, val, None


@pytest.fixture
def benchmark_config() -> TrainingSettings:
    """Load benchmark configuration."""
    config_path = Path(
        "experiments/training/dummy_experiments/dummy_fastmri_knee_reconstruction.yaml"
    )

    if not config_path.exists():
        pytest.skip(f"Benchmark config not found: {config_path}")

    return TrainingSettings.from_yaml(str(config_path))


class TestDataLoaderThroughput:
    """Benchmark DataLoader throughput."""

    def test_baseline_throughput(self, benchmark_config):
        """Measure baseline DataLoader throughput (samples/sec)."""
        # Create DataLoader
        director = DataPipelineDirector(BuilderContext(config=benchmark_config))
        train_loader, _ = director.build_dataloaders()
        _ = None  # the director builds train+val; there is no third loader

        # Warm up (load first batch)
        for batch in train_loader:
            break

        # Benchmark throughput
        num_batches = min(50, len(train_loader))
        batch_size = benchmark_config.data.loader.batch_size

        start_time = time.perf_counter()

        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break

        elapsed = time.perf_counter() - start_time

        # Calculate metrics
        total_samples = num_batches * batch_size
        throughput = total_samples / elapsed
        avg_batch_time = elapsed / num_batches

        print(f"\n{'=' * 60}")
        print("DataLoader Throughput Benchmark")
        print(f"{'=' * 60}")
        print(f"Batch size:       {batch_size}")
        print(f"Batches tested:   {num_batches}")
        print(f"Total samples:    {total_samples}")
        print(f"Total time:       {elapsed:.2f}s")
        print(f"Throughput:       {throughput:.1f} samples/sec")
        print(f"Avg batch time:   {avg_batch_time * 1000:.1f}ms")
        print(f"{'=' * 60}\n")

        # Assertions (minimum performance thresholds)
        assert throughput > 10, f"Throughput too low: {throughput:.1f} samples/sec"
        assert (
            avg_batch_time < 1.0
        ), f"Batch time too high: {avg_batch_time * 1000:.1f}ms"

    def test_pin_memory_impact(self, benchmark_config):
        """Measure pin_memory impact on throughput."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping pin_memory test")


        # Test with pin_memory=True
        original_pin_memory = benchmark_config.data.loader.pin_memory
        benchmark_config.data.loader.pin_memory = True

        train_loader_pinned, _, _ = _build_loaders(benchmark_config)

        # Warm up
        for batch in train_loader_pinned:
            break

        # Benchmark with pin_memory
        num_batches = min(30, len(train_loader_pinned))
        start_time = time.perf_counter()

        for i, batch in enumerate(train_loader_pinned):
            if i >= num_batches:
                break
            # Transfer to GPU
            if isinstance(batch, dict):
                _ = {
                    k: (
                        v.to("cuda", non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

        elapsed_pinned = time.perf_counter() - start_time

        # Test without pin_memory
        benchmark_config.data.loader.pin_memory = False
        train_loader_unpinned, _, _ = _build_loaders(benchmark_config)

        # Warm up
        for batch in train_loader_unpinned:
            break

        # Benchmark without pin_memory
        start_time = time.perf_counter()

        for i, batch in enumerate(train_loader_unpinned):
            if i >= num_batches:
                break
            # Transfer to GPU
            if isinstance(batch, dict):
                _ = {
                    k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

        elapsed_unpinned = time.perf_counter() - start_time

        # Restore original setting
        benchmark_config.data.loader.pin_memory = original_pin_memory

        # Calculate speedup
        speedup = (elapsed_unpinned / elapsed_pinned - 1) * 100

        print(f"\n{'=' * 60}")
        print("Pin Memory Impact Benchmark")
        print(f"{'=' * 60}")
        print(f"With pin_memory:      {elapsed_pinned:.2f}s")
        print(f"Without pin_memory:   {elapsed_unpinned:.2f}s")
        print(f"Speedup:              {speedup:+.1f}%")
        print(f"{'=' * 60}\n")

        # pin_memory should provide speedup
        assert (
            elapsed_pinned <= elapsed_unpinned
        ), "pin_memory should not slow down loading"

    def test_worker_scaling(self, benchmark_config):
        """Measure impact of num_workers on throughput."""

        worker_counts = [0, 2, 4]
        results = {}

        for num_workers in worker_counts:
            benchmark_config.data.loader.num_workers = num_workers

            train_loader, _, _ = _build_loaders(benchmark_config)

            # Warm up
            for batch in train_loader:
                break

            # Benchmark
            num_batches = min(30, len(train_loader))
            start_time = time.perf_counter()

            for i, batch in enumerate(train_loader):
                if i >= num_batches:
                    break

            elapsed = time.perf_counter() - start_time
            throughput = (num_batches * benchmark_config.data.loader.batch_size) / elapsed

            results[num_workers] = {"elapsed": elapsed, "throughput": throughput}

        print(f"\n{'=' * 60}")
        print("Worker Scaling Benchmark")
        print(f"{'=' * 60}")
        for num_workers, metrics in results.items():
            print(
                f"Workers: {num_workers:2d} | "
                f"Time: {metrics['elapsed']:.2f}s | "
                f"Throughput: {metrics['throughput']:.1f} samples/sec"
            )
        print(f"{'=' * 60}\n")

        # Multi-worker should be faster than single-threaded
        if 4 in results and 0 in results:
            speedup = (results[0]["elapsed"] / results[4]["elapsed"] - 1) * 100
            print(f"4 workers vs 0 workers speedup: {speedup:+.1f}%")
            assert speedup > 0, "Multi-worker loading should be faster"


class TestDataLoaderMemory:
    """Benchmark DataLoader memory usage."""

    def test_persistent_workers_memory(self, benchmark_config):
        """Test persistent_workers memory impact."""
        if benchmark_config.data.loader.num_workers == 0:
            pytest.skip("persistent_workers requires num_workers > 0")


        # Test with persistent_workers=False
        benchmark_config.data.loader.persistent_workers = False
        train_loader, _, _ = _build_loaders(benchmark_config)

        # Load some batches
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break

        # Test with persistent_workers=True
        benchmark_config.data.loader.persistent_workers = True
        train_loader, _, _ = _build_loaders(benchmark_config)

        for i, batch in enumerate(train_loader):
            if i >= 10:
                break

        # Both should complete without memory errors
        assert True, "DataLoader created successfully with both settings"
