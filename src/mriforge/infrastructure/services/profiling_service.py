"""Lightweight profiling utilities used across training flows."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Optional imports used only when available. Keep imports local so module
# can be imported on systems without these optional packages.
try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

try:
    import psutil
except Exception:  # pragma: no cover - optional
    psutil = None

from mriforge.domain.interfaces.i_profiling_service import IProfilingService


@dataclass(slots=True)
class ProfilingEvent:
    """Represents a single timed section that was measured."""

    name: str
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ProfilingService(IProfilingService):
    """Collects timing information for hot paths in a thread-safe way."""

    def __init__(self, enabled: bool = False) -> None:
        """__init__.

        Args:
            enabled (bool): Description.
        """
        self._enabled = enabled
        self._events: list[ProfilingEvent] = []
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._start_memory: int | None = None
        # Detect CUDA availability and cache for quick checks. Allow import
        # failures (no torch installed) and fallback to False.
        try:
            self._cuda_available = False
            if torch is not None:
                # Use getattr to avoid static-analysis errors if torch has no
                # cuda attribute in trimmed environments.
                cuda_obj = torch.cuda
                if cuda_obj is not None:
                    try:
                        self._cuda_available = bool(cuda_obj.is_available())
                    except Exception:
                        self._cuda_available = False
        except Exception:  # pragma: no cover - defensive
            self._cuda_available = False

    @classmethod
    def from_config(cls, config: Any | None) -> ProfilingService:
        """Instantiate the service using a config object if available."""

        enabled = False
        if config is not None:
            enabled = bool(config.enable_profiling)
        return cls(enabled=enabled)

    # ------------------------------------------------------------------
    # state management helpers
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """enabled.

        Returns:
            bool: Description.
        """
        return self._enabled

    def enable(self) -> None:
        """enable."""
        self._enabled = True

    def disable(self) -> None:
        """disable."""
        self._enabled = False
        self.reset()

    def reset(self) -> None:
        """Clear any collected profiling events."""

        if not self._events:
            return
        with self._lock:
            self._events.clear()

    # ------------------------------------------------------------------
    # collection helpers
    # ------------------------------------------------------------------
    @contextmanager
    def profile(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Context manager that records execution time for the block."""

        if not self._enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            event = ProfilingEvent(name=name, duration_s=duration, metadata=metadata or {})
            with self._lock:
                self._events.append(event)

    def record_event(
        self,
        name: str,
        *,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Manually append a completed event if needed."""

        if not self._enabled:
            return
        event = ProfilingEvent(name=name, duration_s=duration_s, metadata=metadata or {})
        with self._lock:
            self._events.append(event)

    # ------------------------------------------------------------------
    # reporting helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> list[ProfilingEvent]:
        """Return a copy of the currently buffered events without clearing."""

        with self._lock:
            return list(self._events)

    def flush_metrics(self, *, prefix: str = "profile") -> dict[str, float]:
        """Aggregate collected events into numeric metrics and clear buffer."""

        if not self._enabled:
            self.reset()
            return {}

        with self._lock:
            if not self._events:
                return {}
            events = list(self._events)
            self._events.clear()

        grouped: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total": 0.0, "count": 0.0, "max": 0.0},
        )
        for event in events:
            bucket = grouped[event.name]
            bucket["total"] += event.duration_s
            bucket["count"] += 1.0
            bucket["max"] = max(bucket["max"], event.duration_s)

        metrics: dict[str, float] = {}
        for name, stats in grouped.items():
            key_base = f"{prefix}.{name}"
            total_ms = stats["total"] * 1000.0
            metrics[f"{key_base}.total_ms"] = total_ms
            metrics[f"{key_base}.count"] = stats["count"]
            if stats["count"]:
                metrics[f"{key_base}.avg_ms"] = total_ms / stats["count"]
                metrics[f"{key_base}.max_ms"] = stats["max"] * 1000.0
            else:
                metrics[f"{key_base}.avg_ms"] = 0.0
                metrics[f"{key_base}.max_ms"] = 0.0

        return metrics

    def extend_metrics(
        self,
        target: dict[str, float],
        *,
        prefix: str = "profile",
    ) -> dict[str, float]:
        """Extend the provided target dict in-place with profiling metrics.

        This implements the abstract method declared on IProfilingService. The
        implementation simply flushes current profiling metrics and merges
        them into the provided target dictionary.
        """

        metrics = self.flush_metrics(prefix=prefix)
        # Merge metrics into target in-place
        target.update(metrics)
        return target

    def start_profiling(self) -> None:
        """Start profiling session."""
        if not self._enabled:
            return
        self._start_time = time.perf_counter()
        if self._cuda_available:
            try:
                cuda_obj = torch.cuda
                if cuda_obj is not None:
                    cuda_obj.reset_peak_memory_stats()
                    self._start_memory = cuda_obj.memory_allocated()
                else:
                    self._start_memory = 0
            except Exception:
                self._start_memory = 0
        else:
            try:
                if psutil is not None:
                    self._start_memory = psutil.Process().memory_info().rss
                else:
                    self._start_memory = 0
            except Exception:
                self._start_memory = 0

    def end_profiling(self, name: str) -> dict[str, Any]:
        """End profiling session and return metrics."""
        if not self._enabled or self._start_time is None:
            return {}

        end_time = time.perf_counter()
        duration = end_time - self._start_time

        if self._cuda_available:
            try:
                cuda_obj = torch.cuda
                if cuda_obj is not None:
                    end_memory = cuda_obj.memory_allocated()
                    peak_memory = cuda_obj.max_memory_allocated()
                    memory_used = end_memory - (self._start_memory or 0)
                else:
                    memory_used = 0
                    peak_memory = 0
            except Exception:
                memory_used = 0
                peak_memory = 0
        else:
            try:
                if psutil is not None:
                    end_memory = psutil.Process().memory_info().rss
                    memory_used = end_memory - (self._start_memory or 0)
                    peak_memory = memory_used
                else:
                    memory_used = 0
                    peak_memory = 0
            except Exception:
                memory_used = 0
                peak_memory = 0

        # Reset for next session
        self._start_time = None
        self._start_memory = None

        return {
            "operation": name,
            "duration_seconds": duration,
            "memory_used_mb": memory_used / 1024 / 1024,
            "peak_memory_mb": peak_memory / 1024 / 1024,
            "cuda_available": self._cuda_available,
        }
