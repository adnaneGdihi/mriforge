"""Tests for ``ProfilingService``.

Targets ``spectramr.infrastructure.services.profiling_service``. Lightweight
context-manager-based profiler that aggregates events into latency
metrics for hot paths.

Categories:

- Construction: starts disabled by default; ``from_config`` respects
  ``enable_profiling``
- Disabled: profiling block is a no-op (no events recorded)
- Enabled: ``profile`` records duration + name; ``snapshot`` returns a copy
- ``flush_metrics`` aggregates total/count/avg/max with prefix; clears buffer
- ``record_event`` direct insertion; ignored when disabled
- ``extend_metrics`` merges into a target dict in-place
- ``start_profiling`` / ``end_profiling`` returns duration metric dict
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.services.profiling_service import (
    ProfilingEvent,
    ProfilingService,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction_disabled() -> None:
    """``ProfilingService()`` starts disabled."""
    service = ProfilingService()
    assert service.enabled is False


def test_from_config_enables_when_flag_set() -> None:
    """``from_config`` reads ``enable_profiling`` from the config."""
    cfg = SimpleNamespace(enable_profiling=True)
    service = ProfilingService.from_config(cfg)
    assert service.enabled is True


def test_from_config_with_none_returns_disabled() -> None:
    """No config → disabled service."""
    service = ProfilingService.from_config(None)
    assert service.enabled is False


# ---------------------------------------------------------------------------
# Enable / disable / reset
# ---------------------------------------------------------------------------


def test_enable_then_disable_resets_events() -> None:
    """Disabling clears the event buffer."""
    service = ProfilingService(enabled=True)
    service.record_event("warmup", duration_s=0.001)
    assert len(service.snapshot()) == 1
    service.disable()
    assert service.enabled is False
    assert service.snapshot() == []


def test_reset_empties_events() -> None:
    """``reset`` clears collected events even when enabled."""
    service = ProfilingService(enabled=True)
    service.record_event("foo", duration_s=0.001)
    service.reset()
    assert service.snapshot() == []


# ---------------------------------------------------------------------------
# profile context manager
# ---------------------------------------------------------------------------


def test_profile_block_disabled_records_nothing() -> None:
    """A ``with profile(...)`` block on a disabled service is a no-op."""
    service = ProfilingService(enabled=False)
    with service.profile("noop"):
        pass
    assert service.snapshot() == []


def test_profile_block_records_event_when_enabled() -> None:
    """A ``with profile(...)`` block records exactly one event."""
    service = ProfilingService(enabled=True)
    with service.profile("warmup"):
        time.sleep(0.001)
    events = service.snapshot()
    assert len(events) == 1
    assert events[0].name == "warmup"
    assert events[0].duration_s > 0


def test_profile_block_metadata_persists() -> None:
    """``metadata`` keyword is stored on the recorded event."""
    service = ProfilingService(enabled=True)
    with service.profile("step", metadata={"epoch": 1}):
        pass
    assert service.snapshot()[0].metadata == {"epoch": 1}


# ---------------------------------------------------------------------------
# record_event direct
# ---------------------------------------------------------------------------


def test_record_event_disabled_drops() -> None:
    """``record_event`` is silently ignored when disabled."""
    service = ProfilingService(enabled=False)
    service.record_event("foo", duration_s=0.5)
    assert service.snapshot() == []


def test_record_event_enabled_appends() -> None:
    """``record_event`` appends a new entry."""
    service = ProfilingService(enabled=True)
    service.record_event("foo", duration_s=0.42)
    events = service.snapshot()
    assert len(events) == 1
    assert events[0].duration_s == 0.42


# ---------------------------------------------------------------------------
# flush_metrics
# ---------------------------------------------------------------------------


def test_flush_metrics_disabled_returns_empty() -> None:
    """A disabled service flushes to an empty dict."""
    service = ProfilingService(enabled=False)
    assert service.flush_metrics() == {}


def test_flush_metrics_aggregates_per_name() -> None:
    """Multiple events of the same name aggregate into total/count/avg/max."""
    service = ProfilingService(enabled=True)
    service.record_event("step", duration_s=0.10)
    service.record_event("step", duration_s=0.30)
    metrics = service.flush_metrics(prefix="prof")
    assert metrics["prof.step.count"] == 2.0
    assert pytest.approx(metrics["prof.step.total_ms"]) == 400.0
    assert pytest.approx(metrics["prof.step.avg_ms"]) == 200.0
    assert pytest.approx(metrics["prof.step.max_ms"]) == 300.0


def test_flush_metrics_clears_buffer() -> None:
    """``flush_metrics`` empties the in-memory buffer."""
    service = ProfilingService(enabled=True)
    service.record_event("foo", duration_s=0.01)
    service.flush_metrics()
    assert service.snapshot() == []


# ---------------------------------------------------------------------------
# extend_metrics
# ---------------------------------------------------------------------------


def test_extend_metrics_merges_into_target() -> None:
    """``extend_metrics`` updates the provided dict in place."""
    service = ProfilingService(enabled=True)
    service.record_event("foo", duration_s=0.001)
    target = {"existing": 1.0}
    out = service.extend_metrics(target, prefix="prof")
    assert out is target  # same dict object
    assert "existing" in out
    assert "prof.foo.total_ms" in out


# ---------------------------------------------------------------------------
# start/end_profiling session
# ---------------------------------------------------------------------------


def test_start_end_returns_duration() -> None:
    """``start/end`` produces a metrics dict with ``duration_seconds``."""
    service = ProfilingService(enabled=True)
    service.start_profiling()
    time.sleep(0.001)
    metrics = service.end_profiling("session")
    assert metrics["operation"] == "session"
    assert metrics["duration_seconds"] > 0


def test_end_without_start_returns_empty() -> None:
    """``end_profiling`` with no prior ``start`` returns empty dict."""
    service = ProfilingService(enabled=True)
    assert service.end_profiling("session") == {}


# ---------------------------------------------------------------------------
# ProfilingEvent dataclass
# ---------------------------------------------------------------------------


def test_event_default_metadata_factory() -> None:
    """``metadata`` defaults to an empty dict via field factory."""
    event = ProfilingEvent(name="test", duration_s=0.0)
    assert event.metadata == {}
