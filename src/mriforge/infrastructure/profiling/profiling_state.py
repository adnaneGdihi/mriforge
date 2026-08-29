"""Profiling State Module
======================

This module provides the ProfilingState class which acts as a unified facade
for NVTX profiling and memory tracking, suitable for integration into the
TrainingState.
"""

import logging
from contextlib import contextmanager
from typing import Any

from mriforge.models.profiling.advanced_profiling import (
    MemoryTracker,
    NVTXProfiler,
    ProfilingConfig,
)

logger = logging.getLogger(__name__)


class ProfilingState:
    """Unified profiling state management.

    Acts as a facade/adapter over NVTXProfiler and MemoryTracker to provide
    a simple interface for the training loop.
    """

    def __init__(self, config: ProfilingConfig | None = None):
        """Initialize profiling state.

        Args:
            config: Profiling configuration. If None, default config is used.
        """
        self.config = config or ProfilingConfig()
        self.profiler = NVTXProfiler(config=self.config)
        self.memory_tracker = MemoryTracker(config=self.config)
        self.enabled = self.config.enable_profiler

        logger.info(f"ProfilingState initialized (enabled={self.enabled})")

    @contextmanager
    def profile(self, name: str):
        """Profile a block of code with NVTX range."""
        with self.profiler.profile_range(name):
            yield

    def start(self):
        """Required for API compatibility if needed."""
        pass

    def stop(self):
        """Required for API compatibility if needed."""
        pass

    def step(self):
        """Perform step-wise profiling actions (e.g. memory sampling)."""
        if self.config.track_memory:
            self.memory_tracker.record_memory_sample()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of profiling stats."""
        return self.memory_tracker.get_memory_report()
