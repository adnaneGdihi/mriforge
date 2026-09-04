"""Fail-Fast Error Handling Service

Simplified error handling service that logs errors and re-raises them immediately.
Implements fail-fast architecture - no recovery attempts or fallbacks.
"""

import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from spectramr.domain.interfaces.service_interfaces import (
    ILoggingService,
    IMetricsService,
)
from spectramr.infrastructure.di.di_container import resolve_service


# Define IErrorHandlingService interface locally since it's not in service_interfaces
class IErrorHandlingService:
    """Interface for error handling services."""

    def handle_error(self, error: Exception, context: dict | None = None) -> None:
        """Handle an error.

        Args:
            error: The exception to handle
            context: Optional context dictionary
        """
        raise NotImplementedError


@dataclass
class ErrorRecord:
    """Record of a handled error."""

    error_type: str
    error_message: str
    context: dict[str, Any]
    timestamp: str
    severity: str
    stack_trace: str


@dataclass
class ErrorStatistics:
    """Statistics about error handling."""

    total_errors: int = 0
    errors_by_type: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int),
    )
    errors_by_severity: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int),
    )
    recent_errors: list[ErrorRecord] = field(default_factory=list)


class FailFastErrorHandlingService(IErrorHandlingService):
    """Fail-fast error handling service.

    Logs errors and re-raises them immediately. No recovery attempts.

    ### Fail-Fast Error Path
    ```mermaid
    sequenceDiagram
        participant C as Caller (Pipeline/Strategy)
        participant S as Error Service
        participant L as Logging
        participant M as Metrics

        C->>S: handle_error(exc, context)
        S->S: create_error_record()
        S->S: update_statistics(type, severity)
        S->>L: log(severity, msg)
        S-->>C: RAISE exc (abort immediately)
    ```
    """

    def __init__(
        self,
        logging_service: ILoggingService | None = None,
        metrics_service: IMetricsService | None = None,
    ):
        """Initialize the error handling service.

        Args:
            logging_service: Service for logging errors
            metrics_service: Service for tracking error metrics
        """
        self._statistics = ErrorStatistics()
        self._max_recent_errors = 100

        # Resolve dependencies - fail fast if critical services not found
        self._logging_service = logging_service or resolve_service(ILoggingService)
        self._metrics_service = metrics_service or resolve_service(IMetricsService)
        self._logger = logging.getLogger(__name__)

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "medium",
    ) -> dict[str, Any]:
        """Handle an error by logging it and re-raising immediately.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Severity level ('low', 'medium', 'high', 'critical')

        Returns:
            Dictionary with handling results

        Raises:
            The original exception after logging
        """
        context = context or {}
        error_type = type(error).__name__
        error_message = str(error)
        timestamp = datetime.now().isoformat()

        # Create error record
        error_record = ErrorRecord(
            error_type=error_type,
            error_message=error_message,
            context=context,
            timestamp=timestamp,
            severity=severity,
            stack_trace=traceback.format_exc(),
        )

        # Update statistics
        self._update_statistics(error_record)

        # Log the error
        self._log_error(error, context, severity)

        # Error counters live in `_statistics` (updated above), NOT in a metrics
        # counter. There used to be two `self._metrics_service.increment_counter(...)`
        # calls here recording `error_handling.<type>` and `error_handling.total`
        # with a severity tag (#708). `increment_counter` is defined NOWHERE in the
        # repository -- not on `MetricsTracker`, not on `IMetricsService` -- so every
        # call raised `AttributeError` into the `except Exception` below and was
        # downgraded to a warning. The telemetry was never recorded, and the
        # swallow is what kept that invisible for as long as it existed.
        #
        # Deleted rather than implemented: `_update_statistics` already records
        # `total_errors`, `errors_by_type` and `errors_by_severity` -- exactly the
        # two counters, by exactly the same keys. Adding the method would have
        # duplicated working state into a surface that is itself nearly all dead
        # (#710). Nothing is lost: read `get_statistics()`.

        # Store recent error
        self._statistics.recent_errors.append(error_record)
        if len(self._statistics.recent_errors) > self._max_recent_errors:
            self._statistics.recent_errors.pop(0)

        # Re-raise the error (fail-fast)
        raise error

    def get_error_statistics(self) -> dict[str, Any]:
        """Get statistics about handled errors.

        Returns:
            Dictionary with error statistics
        """
        return {
            "total_errors": self._statistics.total_errors,
            "errors_by_type": dict(self._statistics.errors_by_type),
            "errors_by_severity": dict(self._statistics.errors_by_severity),
            "recent_errors_count": len(self._statistics.recent_errors),
        }

    def create_error_context(
        self,
        operation: str,
        component: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized error context.

        Args:
            operation: The operation being performed
            component: The component where the error occurred
            metadata: Additional metadata

        Returns:
            Standardized error context dictionary
        """
        return {
            "operation": operation,
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def _update_statistics(self, error_record: ErrorRecord) -> None:
        """Update error statistics."""
        self._statistics.total_errors += 1
        self._statistics.errors_by_type[error_record.error_type] += 1
        self._statistics.errors_by_severity[error_record.severity] += 1

    def _log_error(
        self,
        error: Exception,
        context: dict[str, Any],
        severity: str,
    ) -> None:
        """Log an error with appropriate level."""
        error_msg = f"Error in {context.get('component', 'unknown')}: {error}"
        if context.get("operation"):
            error_msg += f" during {context['operation']}"

        log_level = {
            "low": "DEBUG",
            "medium": "WARNING",
            "high": "ERROR",
            "critical": "CRITICAL",
        }.get(severity, "ERROR")

        if self._logging_service:
            self._logging_service.log(log_level, error_msg)
        else:
            self._logger.log(getattr(logging, log_level), error_msg)
