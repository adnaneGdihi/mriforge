r"""End-of-experiment reporting pipeline.

Per-experiment figure + table generation aligned with the design
specified in `TODO/report_step` (Phases 0-6, Parts A/B/C). Driven by
`generate_report(experiment_dir, ...)` and called automatically from
`mriforge.pipelines.train.run_training_pipeline` when
``ReportingSettings.enabled`` is True.
"""

from .aggregator import aggregate, aggregate_many, discover_artifacts
from .batch import discover_run_dirs, generate_reports
from .metadata import RunMetadata, detect_metadata, stamp_figure, write_sidecar
from .pipeline import TASK_PRESETS, generate_report
from .style import METHOD_COLOURS, OKABE_ITO, colour_for, use_default_style

__all__ = [
    "METHOD_COLOURS",
    "OKABE_ITO",
    "TASK_PRESETS",
    "RunMetadata",
    "aggregate",
    "aggregate_many",
    "colour_for",
    "detect_metadata",
    "discover_artifacts",
    "discover_run_dirs",
    "generate_report",
    "generate_reports",
    "stamp_figure",
    "use_default_style",
    "write_sidecar",
]
