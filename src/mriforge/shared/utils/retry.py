"""Retry and backoff utilities.

Provides simple helpers to compute exponential backoff delays and to sleep with
those delays in a centralized, testable way.
"""

from __future__ import annotations

import random
import time


def compute_backoff_seconds(
    attempt: int,
    base_seconds: float = 1.0,
    factor: float = 2.0,
    max_seconds: float | None = None,
    jitter: bool = True,
) -> float:
    """Compute exponential backoff delay in seconds with optional jitter.

    Args:
        attempt: Current attempt number (1-based preferred, but 0 works).
        base_seconds: Base delay in seconds for the first attempt.
        factor: Multiplicative factor for exponential growth.
        max_seconds: Maximum cap for the delay.
        jitter: Whether to add random jitter to prevent thundering herd.

    Returns:
        Delay in seconds (capped, with jitter if enabled).

    """
    a = max(0, attempt)
    delay = base_seconds * (factor ** max(a - 1, 0))
    if max_seconds is not None:
        delay = min(delay, max_seconds)

    if jitter and delay > 0:
        # Add jitter: randomize delay between 0.5x and 1.5x the computed delay
        jitter_factor = 0.5 + random.random()  # Random between 0.5 and 1.5
        delay *= jitter_factor

    return delay


def sleep_with_backoff(
    attempt: int,
    base_seconds: float = 1.0,
    factor: float = 2.0,
    max_seconds: float | None = None,
    jitter: bool = True,
) -> float:
    """Sleep using exponential backoff with jitter and return the actual delay used.

    Args are the same as compute_backoff_seconds(). The function sleeps for the
    computed duration (with jitter by default) and returns that value for
    observability/testing.
    """
    delay = compute_backoff_seconds(
        attempt=attempt,
        base_seconds=base_seconds,
        factor=factor,
        max_seconds=max_seconds,
        jitter=jitter,
    )
    if delay > 0:
        time.sleep(delay)
    return delay
