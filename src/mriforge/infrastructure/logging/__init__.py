"""Logging Infrastructure Package.

Canonical import path for all logging functionality. Re-exports from
their current source locations to provide a unified public API:

- ``LoggingService`` / ``ComprehensiveLoggingService`` — structured text + TensorBoard
- ``MetricsTracker`` — CSV metrics, validation image saving (``IMetricsService``)
- ``TrainingLogDelegate`` — async non-blocking loss CSV writer
- ``LoggingServiceFactory`` — factory for config-driven logger creation

Usage::

    from mriforge.infrastructure.logging import LoggingService, MetricsTracker

Note: Uses lazy ``__getattr__`` to avoid circular imports with
``mriforge.infrastructure.services.__init__`` which also re-exports LoggingService.
"""

from .banners import banner, phase  # re-exported via __all__
from .progress import smart_progress  # re-exported via __all__
from .provenance import (
    collect_run_provenance,
    count_parameters,
    cpu_resources,
    format_provenance_lines,
    git_provenance,
    gpu_resources,
    log_provenance,
    memory_resources,
    node_resources,
    runtime_environment,
    torch_runtime,
)


def __getattr__(name: str):
    """Lazy imports to break circular dependency with services.__init__."""
    if name in (
        "LoggingService",
        "ComprehensiveLoggingService",
        "LoggingServiceFactory",
        "create_logging_service",
    ):
        from mriforge.infrastructure.services.logging_service import (
            ComprehensiveLoggingService,
            LoggingService,
            LoggingServiceFactory,
            create_logging_service,
        )

        _exports = {
            "LoggingService": LoggingService,
            "ComprehensiveLoggingService": ComprehensiveLoggingService,
            "LoggingServiceFactory": LoggingServiceFactory,
            "create_logging_service": create_logging_service,
        }
        return _exports[name]

    if name == "MetricsTracker":
        from mriforge.infrastructure.services.metrics_tracker import MetricsTracker

        return MetricsTracker

    if name == "TrainingLogDelegate":
        from mriforge.infrastructure.services.persistence.training_log_delegate import (
            TrainingLogDelegate,
        )

        return TrainingLogDelegate

    if name == "ColoredConsoleFormatter":
        from mriforge.infrastructure.services.logging_service import (
            ColoredConsoleFormatter,
        )

        return ColoredConsoleFormatter

    if name == "bootstrap_console_logging":
        from mriforge.infrastructure.services.logging_service import (
            bootstrap_console_logging,
        )

        return bootstrap_console_logging

    raise AttributeError(f"module 'mriforge.infrastructure.logging' has no attribute {name!r}")


# A SINGLE ``__all__`` — the union of the eagerly imported re-exports (banners /
# progress / provenance) and the lazily exposed services (resolved through
# ``__getattr__`` above). There used to be TWO ``__all__`` assignments; the
# second silently clobbered the first, dropping every eager name from
# ``from ... import *`` (#6 of the 2026-06-19 logging-duplication audit).
__all__ = [
    "ColoredConsoleFormatter",
    "ComprehensiveLoggingService",
    "LoggingService",
    "LoggingServiceFactory",
    "MetricsTracker",
    "TrainingLogDelegate",
    "banner",
    "bootstrap_console_logging",
    "collect_run_provenance",
    "count_parameters",
    "cpu_resources",
    "create_logging_service",
    "format_provenance_lines",
    "git_provenance",
    "gpu_resources",
    "log_provenance",
    "memory_resources",
    "node_resources",
    "phase",
    "runtime_environment",
    "smart_progress",
    "torch_runtime",
]
