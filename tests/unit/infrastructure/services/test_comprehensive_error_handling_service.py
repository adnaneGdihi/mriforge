"""Tests for the error-handling service's statistics, and for #708.

The service used to call ``self._metrics_service.increment_counter(...)`` twice per
handled error, recording ``error_handling.<type>`` and ``error_handling.total``.
``increment_counter`` is defined NOWHERE in the repository -- not on
``MetricsTracker``, not on ``IMetricsService`` -- so every call raised
``AttributeError`` into a bare ``except Exception`` and was downgraded to a warning.
The telemetry was never recorded, and the swallow is what kept that invisible.

The calls are deleted rather than implemented: ``_update_statistics`` already records
``total_errors``, ``errors_by_type`` and ``errors_by_severity`` -- the same two
counters under the same keys -- so implementing the method would have duplicated
working state into a surface that is itself nearly all dead (#710).
"""

from __future__ import annotations

import inspect

import pytest

from spectramr.infrastructure.services import comprehensive_error_handling_service as ehs


class TestNoIncrementCounterCall:
    def test_increment_counter_is_defined_nowhere(self):
        """The premise. If someone later implements it, this test says so."""
        from spectramr.domain.interfaces.service_interfaces import IMetricsService
        from spectramr.infrastructure.services.metrics_tracker import MetricsTracker

        assert not hasattr(MetricsTracker, "increment_counter")
        assert not hasattr(IMetricsService, "increment_counter")

    def test_the_service_no_longer_calls_it(self):
        """Source-level: the defect was a call that ALWAYS raised into a swallow.

        A behavioural test cannot see it — the `except Exception` made a raising
        call and a working one produce identical observable behaviour, which is
        precisely why it survived.
        """
        src = inspect.getsource(ehs)
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "increment_counter" not in code


class TestErrorStatisticsStillRecordTheSameCounters:
    """What the deleted calls were *trying* to record is already recorded."""

    @staticmethod
    def _service():
        logging_service = _StubLogging()
        return ehs.FailFastErrorHandlingService(
            logging_service=logging_service, metrics_service=_StubMetrics()
        )

    def test_totals_and_breakdowns_accumulate(self):
        service = self._service()
        for exc, sev in (
            (ValueError("a"), "error"),
            (ValueError("b"), "error"),
            (KeyError("c"), "warning"),
        ):
            try:
                service.handle_error(exc, {"component": "t"}, severity=sev)
            except Exception:
                # A fail-fast service may re-raise; the statistics update happens
                # before that, which is the behaviour under test.
                pass

        stats = service._statistics
        assert stats.total_errors == 3
        assert stats.errors_by_type["ValueError"] == 2
        assert stats.errors_by_type["KeyError"] == 1
        assert stats.errors_by_severity["error"] == 2
        assert stats.errors_by_severity["warning"] == 1

    def test_the_metrics_service_is_never_touched_for_counters(self):
        """It is still injected (other code resolves it); it just is not used here."""
        metrics = _StubMetrics()
        service = ehs.FailFastErrorHandlingService(
            logging_service=_StubLogging(), metrics_service=metrics
        )
        try:
            service.handle_error(ValueError("x"), {"component": "t"}, severity="error")
        except Exception:
            pass
        assert metrics.calls == [], f"unexpected metrics-service calls: {metrics.calls}"


class _StubLogging:
    def __init__(self):
        self.records = []

    def log(self, level, message, *a, **k):
        self.records.append((level, message))

    def log_error(self, message, *a, **k):
        self.records.append(("error", message))

    def log_warning(self, message, *a, **k):
        self.records.append(("warning", message))

    def log_info(self, message, *a, **k):
        self.records.append(("info", message))

    def log_critical(self, message, *a, **k):
        self.records.append(("critical", message))


class _StubMetrics:
    """Records ANY attribute access, so a reintroduced counter call is visible."""

    def __init__(self):
        self.calls: list[str] = []

    def __getattr__(self, name):
        def _recorder(*args, **kwargs):
            self.calls.append(name)

        self.calls.append(f"<attr:{name}>")
        return _recorder


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
