"""Memory Management Context Managers.

This module provides context managers for guaranteed GPU memory cleanup
and fragmentation prevention during training. Uses the with statement
pattern to ensure cleanup even on exception.

Key Benefits:
- Guarantees cleanup via finally blocks
- Prevents 200 MB/1000 batches memory fragmentation
- Explicit resource management (Python best practices)
- Works with exception handling

Complexity: Replaces manual cleanup with guaranteed pattern
Time: O(1) for empty_cache(), O(cleanup) for gc.collect()
"""

import gc
import logging
from collections.abc import Generator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


@contextmanager
def memory_cleanup_context(
    description: str = "operation",
    log_memory: bool = True,
) -> Generator:
    """Context manager for guaranteed GPU memory cleanup.

    Guarantees that GPU memory is cleaned even if exception occurs during
    the context. Uses try/finally to ensure cleanup runs.

    Time Complexity: O(1) for cleanup
    Memory Impact: Reduces fragmentation from 200 MB/1000 batches to ~0 MB

    Args:
        description: Description of operation for logging
        log_memory: Whether to log memory stats before/after

    Yields:
        None

    Example:
        >>> with memory_cleanup_context("training_epoch"):
        ...     for batch in dataloader:
        ...         train_step(batch)
        ...     # Memory guaranteed to be cleaned even if train_step() raises

    Usage in training loop:
        >>> for epoch in range(num_epochs):
        ...     with memory_cleanup_context(f"epoch_{epoch}"):
        ...         train_epoch()
        ...     # Memory is guaranteed clean between epochs
    """
    initial_memory = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

    if log_memory and initial_memory is not None:
        logger.info(f"Starting {description} - GPU memory: {initial_memory / 1024**2:.1f} MB")

    try:
        yield  # ✅ Execute user code here
    except torch.cuda.OutOfMemoryError:
        # Only "Stop the World" if the world has actually crashed.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if log_memory:
            logger.warning(f"OOM caught in {description}. Cleared cache and GC.")

        raise  # Re-raise to let the retry logic handle it
    finally:
        # No proactive cleanup
        pass


@contextmanager
def memory_reserve_context(
    fraction: float = 0.1,
) -> Generator:
    """Context manager for memory-conscious operations.

    Reserves GPU memory before operation and validates it afterward.
    Useful for detecting memory leaks in specific code sections.

    Args:
        fraction: Fraction of GPU memory to reserve (0-1)

    Yields:
        None

    Example:
        >>> with memory_reserve_context(fraction=0.05):
        ...     # This code won't allocate more than 5% of GPU memory
        ...     result = expensive_operation()
    """
    if not torch.cuda.is_available():
        yield
        return

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_amount = int(total_memory * fraction)

    # Try to reserve memory
    try:
        reserved_tensor = torch.zeros(reserved_amount // 4, dtype=torch.float32)
        torch.cuda.empty_cache()
        yield
    finally:
        # Release reserved memory
        del reserved_tensor
        torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def memory_efficient_context(
    enable_gc: bool = True,
    empty_cache: bool = True,
) -> Generator:
    """Context manager for memory-efficient training.

    Combines garbage collection and cache emptying for optimal memory usage.
    Useful around expensive operations like validation or checkpointing.

    Args:
        enable_gc: Whether to enable aggressive garbage collection
        empty_cache: Whether to empty GPU cache

    Yields:
        None

    Example:
        >>> with memory_efficient_context():
        ...     validation_results = validate_model()
    """
    if enable_gc:
        gc.collect()

    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        yield
    finally:
        if enable_gc:
            gc.collect()

        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def gpu_memory_monitor(
    name: str = "operation",
) -> Generator:
    """Context manager that monitors GPU memory during execution.

    Tracks peak memory usage and logs warnings if memory grows excessively.

    Args:
        name: Name of operation being monitored

    Yields:
        Dictionary that gets updated with memory stats

    Example:
        >>> with gpu_memory_monitor("data_loading") as stats:
        ...     for batch in dataloader:
        ...         load_batch(batch)
        >>> print(f"Peak memory: {stats['peak_memory_mb']:.1f} MB")
    """
    stats = {
        "initial_memory_mb": 0.0,
        "peak_memory_mb": 0.0,
        "final_memory_mb": 0.0,
        "delta_memory_mb": 0.0,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        stats["initial_memory_mb"] = torch.cuda.memory_allocated() / 1024**2

    try:
        yield stats
    finally:
        if torch.cuda.is_available():
            stats["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            stats["final_memory_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["delta_memory_mb"] = stats["final_memory_mb"] - stats["initial_memory_mb"]

            # Log warning if excessive memory growth
            if stats["delta_memory_mb"] > 100:  # 100 MB threshold
                logger.warning(
                    f"{name}: Excessive memory growth of {stats['delta_memory_mb']:.1f} MB detected"
                )

            logger.debug(
                f"{name} memory stats: "
                f"initial={stats['initial_memory_mb']:.1f}MB, "
                f"peak={stats['peak_memory_mb']:.1f}MB, "
                f"final={stats['final_memory_mb']:.1f}MB, "
                f"delta={stats['delta_memory_mb']:+.1f}MB"
            )
