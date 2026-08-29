"""Phase 4b-3: Advanced optimization utilities for performance enhancement.

This module provides infrastructure for:
1. Gradient buffer caching - Cache gradients across passes
2. Async metrics reporting - Non-blocking metric collection
3. Performance tracking - Optimization validation
"""

import logging
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class GradientBuffer:
    """Phase 4b-3: Cache for gradient tensors across multiple passes.

    Purpose: Reduce redundant backward pass computations by caching
    gradients computed in discriminator training for reuse in generator
    training phase.

    Features:
    - Weak references to prevent memory leaks
    - Thread-safe access with locks
    - Automatic cache eviction on tensor release
    - Performance tracking (hit rate, memory)

    Example:
        buffer = GradientBuffer(capacity=100)
        buffer.store("d_grad_1", d_loss_gradients)
        d_grads = buffer.retrieve("d_grad_1")  # Returns cached gradients
    """

    capacity: int = 100
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _cache: dict[str, weakref.ref] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
        },
        init=False,
    )

    def store(self, key: str, gradients: torch.Tensor) -> None:
        """Store gradient tensor in cache with weak reference.

        Args:
            key: Unique identifier for gradients (e.g., "d_grad_real")
            gradients: Tensor containing gradients to cache
        """
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.capacity:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            # Store weak reference to gradient tensor
            self._cache[key] = weakref.ref(gradients)
            self._stats["stores"] += 1

    def retrieve(self, key: str) -> torch.Tensor | None:
        """Retrieve cached gradients by key.

        Args:
            key: Identifier for cached gradients

        Returns:
            Cached tensor if present and still valid, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            # Check if weak reference is still valid
            grad_ref = self._cache[key]
            gradients = grad_ref()

            if gradients is None:
                # Tensor was garbage collected
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return gradients

    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Float in [0, 1] representing hit / (hit + miss) ratio
        """
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dictionary with hits, misses, stores, evictions, hit_rate
        """
        return {
            **self._stats,
            "hit_rate": self.hit_rate(),
            "current_size": len(self._cache),
            "capacity": self.capacity,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "stores": 0,
                "evictions": 0,
            }


@dataclass
class AsyncMetricsReporter:
    """Phase 4b-3: Non-blocking metrics reporting for training loop.

    Purpose: Collect metrics asynchronously to avoid blocking training steps
    with logging I/O and GPU-CPU synchronization.

    Features:
    - ThreadPoolExecutor for non-blocking reporting
    - Bounded queue to prevent memory buildup
    - Automatic batch flushing
    - Type-safe metric handling

    Example:
        reporter = AsyncMetricsReporter(batch_size=10)
        reporter.report_async({"loss": 0.5, "accuracy": 0.95})
        reporter.flush()  # Wait for pending metrics to complete
    """

    batch_size: int = 10
    queue_size: int = 100
    num_workers: int = 1
    _queue: Queue = field(default_factory=lambda: Queue(maxsize=100), init=False)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1), init=False
    )
    _pending_futures: list = field(default_factory=list, init=False)
    _batch: list = field(default_factory=list, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _callback: Callable[[dict[str, float]], None] | None = field(default=None, init=False)

    def __post_init__(self):
        """Initialize queue with configured size."""
        self._queue = Queue(maxsize=self.queue_size)
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def set_callback(self, callback: Callable[[dict[str, float]], None]) -> None:
        """Set callback function for metric reporting.

        Args:
            callback: Function to call with each batch of metrics
        """
        self._callback = callback

    def report_async(self, metrics: dict[str, float]) -> None:
        """Queue metrics for asynchronous reporting.

        Args:
            metrics: Dictionary of metric names to float values
        """
        try:
            self._queue.put_nowait(metrics)
        except Full:
            # Queue is full, drop oldest batch to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(metrics)
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)  # Silently drop if still full

        # Check if batch is ready to process
        with self._lock:
            self._batch.append(metrics)
            if len(self._batch) >= self.batch_size:
                self._submit_batch()

    def _submit_batch(self) -> None:
        """Submit current batch for processing in executor.

        Must be called with _lock held.
        """
        if not self._batch or self._callback is None:
            return

        batch_to_process = self._batch.copy()
        self._batch.clear()

        # Aggregate metrics across batch
        aggregated = self._aggregate_metrics(batch_to_process)

        # Submit to executor
        future = self._executor.submit(self._callback, aggregated)
        self._pending_futures.append(future)

        # Clean up completed futures
        self._pending_futures[:] = [f for f in self._pending_futures if not f.done()]

    def _aggregate_metrics(self, batch: list) -> dict[str, float]:
        """Aggregate metrics across batch by averaging.

        Args:
            batch: List of metric dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        if not batch:
            return {}

        aggregated: dict[str, list] = {}
        for metrics in batch:
            for key, value in metrics.items():
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(value)

        # Average all values
        return {key: sum(vals) / len(vals) for key, vals in aggregated.items()}

    def flush(self) -> None:
        """Flush pending metrics and wait for completion.

        Blocks until all pending metrics are reported.
        """
        with self._lock:
            if self._batch and self._callback is not None:
                self._submit_batch()

        # Wait for all futures to complete
        for future in self._pending_futures:
            try:
                future.result(timeout=5.0)
            except Exception as _exc:
                logger.debug("Suppressed exception: %s", _exc)

        self._pending_futures.clear()

    def shutdown(self) -> None:
        """Shutdown executor and flush pending metrics."""
        self.flush()
        self._executor.shutdown(wait=True)


@dataclass
class OptimizationMetrics:
    """Track Phase 4b-3 optimization metrics for validation.

    Measures:
    - Time per training step
    - Generator inference time
    - Discriminator training time
    - Backward pass time
    - Metrics logging time
    - GPU-CPU sync points
    """

    name: str = "optimization_metrics"
    _timings: dict[str, list] = field(
        default_factory=lambda: {
            "step_total": [],
            "generator_forward": [],
            "discriminator_train": [],
            "backward_pass": [],
            "metrics_logging": [],
        },
        init=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record_timing(self, component: str, duration_ms: float) -> None:
        """Record timing measurement for a component.

        Args:
            component: Component name (e.g., "generator_forward")
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            if component not in self._timings:
                self._timings[component] = []
            self._timings[component].append(duration_ms)

    def get_stats(self, component: str) -> dict[str, float]:
        """Get statistics for a component's timings.

        Args:
            component: Component name

        Returns:
            Dictionary with mean, median, min, max timings
        """
        with self._lock:
            timings = self._timings.get(component, [])

        if not timings:
            return {}

        timings_sorted = sorted(timings)
        return {
            "mean": sum(timings) / len(timings),
            "median": timings_sorted[len(timings) // 2],
            "min": min(timings),
            "max": max(timings),
            "count": len(timings),
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all components.

        Returns:
            Dictionary mapping components to their stats
        """
        return {component: self.get_stats(component) for component in self._timings.keys()}

    def reset(self) -> None:
        """Reset all timing measurements."""
        with self._lock:
            for component in self._timings:
                self._timings[component].clear()


class PerformanceContextManager:
    """Context manager for timing code blocks during optimization.

    Usage:
        metrics = OptimizationMetrics()
        with PerformanceContextManager("generator_forward", metrics):
            # Code to time
            pass
    """

    def __init__(self, component: str, metrics: OptimizationMetrics):
        """Initialize context manager.

        Args:
            component: Component name for timing
            metrics: OptimizationMetrics instance
        """
        self.component = component
        self.metrics = metrics
        self.start_time = None

    def __enter__(self):
        """Enter context and start timing."""
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Exit context and record timing."""
        import time

        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        self.metrics.record_timing(self.component, duration_ms)
