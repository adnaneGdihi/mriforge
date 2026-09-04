"""Analysis Package
================

Analysis and profiling utilities for the spectraMR project. Following
SOLID principles for maintainable and extensible analysis tools.

The thin ``run_*_training`` wrappers around
:func:`spectramr.pipelines.train.run_training_pipeline` were removed
2026-05-14 — they imported upward from ``pipelines/`` (CLAUDE.md
layer-direction violation, ``TODO/backlog_ssot_and_layering_cleanup.md``
Phase 2) and had zero callers. Callers should now invoke
``run_training_pipeline`` directly from the pipelines layer.
"""

try:
    from .analyze_gradient_logs import analyze_gradient_logs
    from .extensive_gradient_logger import ExtensiveGradientLogger
    from .profile_training import TrainingProfiler
except ImportError as e:
    import logging

    logging.getLogger(__name__).warning("Some analysis modules unavailable: %s", e)

    def analyze_gradient_logs(*args, **kwargs):  # type: ignore[no-redef]
        """Placeholder when optional analysis deps are missing."""
        raise ImportError("analyze_gradient_logs requires the analysis extras")

    class ExtensiveGradientLogger:  # type: ignore[no-redef]
        """Placeholder when optional analysis deps are missing."""

    class TrainingProfiler:  # type: ignore[no-redef]
        """Placeholder when optional analysis deps are missing."""


__all__ = [
    "ExtensiveGradientLogger",
    "TrainingProfiler",
    "analyze_gradient_logs",
]
