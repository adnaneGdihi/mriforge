"""Tests for the retry / backoff utilities.

Targets ``mriforge.shared.utils.retry``. Verifies exponential growth, cap
behaviour, jitter randomisation, and the side-effecting sleep variant.
"""

from __future__ import annotations

import math
import random
import time

import pytest

from mriforge.shared.utils.retry import compute_backoff_seconds, sleep_with_backoff


# ---------------------------------------------------------------------------
# compute_backoff_seconds
# ---------------------------------------------------------------------------


def test_first_attempt_uses_base_delay_no_jitter() -> None:
    """attempt=1 with jitter=False returns exactly ``base_seconds``."""
    delay = compute_backoff_seconds(attempt=1, base_seconds=0.5, factor=2.0, jitter=False)
    assert math.isclose(delay, 0.5)


def test_exponential_growth_no_jitter() -> None:
    """attempt n with factor 2 → base * 2^(n-1)."""
    base = 0.1
    for n, expected in [(1, 0.1), (2, 0.2), (3, 0.4), (4, 0.8)]:
        delay = compute_backoff_seconds(
            attempt=n, base_seconds=base, factor=2.0, jitter=False
        )
        assert math.isclose(delay, expected, rel_tol=1e-9)


def test_max_seconds_caps_delay() -> None:
    """``max_seconds`` enforces an upper bound."""
    delay = compute_backoff_seconds(
        attempt=10, base_seconds=1.0, factor=2.0, max_seconds=5.0, jitter=False
    )
    assert delay == 5.0


def test_zero_attempt_is_clamped_to_zero() -> None:
    """attempt=0 → no delay (the implementation uses ``max(0, attempt)``)."""
    delay = compute_backoff_seconds(attempt=0, base_seconds=1.0, jitter=False)
    assert math.isclose(delay, 1.0)  # base_seconds * 2**0 = base


def test_negative_attempt_treated_as_zero() -> None:
    """Negative attempt values are clamped (defensive)."""
    delay = compute_backoff_seconds(attempt=-5, base_seconds=2.0, jitter=False)
    assert math.isclose(delay, 2.0)


def test_jitter_produces_value_in_expected_range() -> None:
    """With jitter, delay ∈ [0.5x, 1.5x] of nominal — over many trials."""
    random.seed(0)  # reproducibility
    nominal = 1.0
    for _ in range(50):
        delay = compute_backoff_seconds(
            attempt=1, base_seconds=nominal, factor=2.0, jitter=True
        )
        assert 0.5 * nominal <= delay <= 1.5 * nominal


def test_jitter_disabled_is_deterministic() -> None:
    """With jitter=False, two calls with the same args return the same value."""
    a = compute_backoff_seconds(attempt=3, base_seconds=0.3, jitter=False)
    b = compute_backoff_seconds(attempt=3, base_seconds=0.3, jitter=False)
    assert a == b


# ---------------------------------------------------------------------------
# sleep_with_backoff
# ---------------------------------------------------------------------------


def test_sleep_with_backoff_returns_actual_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The function returns the delay it slept for; sleep call is mocked."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    delay = sleep_with_backoff(attempt=1, base_seconds=0.5, jitter=False)
    assert math.isclose(delay, 0.5)
    assert sleeps == [0.5]


def test_sleep_with_backoff_zero_attempt_still_sleeps_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """attempt=0 still produces a base-second sleep (delay > 0)."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    sleep_with_backoff(attempt=0, base_seconds=0.25, jitter=False)
    assert len(sleeps) == 1


def test_sleep_with_backoff_zero_delay_skips_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If computed delay is 0, time.sleep is NOT called."""
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleeps.append(d))
    delay = sleep_with_backoff(
        attempt=1, base_seconds=0.0, jitter=False, max_seconds=0.0
    )
    assert delay == 0.0
    assert sleeps == []
