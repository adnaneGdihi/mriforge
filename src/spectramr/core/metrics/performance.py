"""Performance Metrics Module.

Defines data structures for tracking computational performance metrics
such as training time, memory usage, GPU utilization, and throughput.
"""

from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for computational performance metrics."""

    forward_time: float = 0.0
    backward_time: float = 0.0
    total_time: float = 0.0
    memory_peak: float = 0.0
    memory_allocated: float = 0.0
    memory_reserved: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0  # samples/second
    flops: int = 0
    parameters: int = 0
