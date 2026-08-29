"""The end-of-run reporting hook, shared by every verb that produces a run dir.

This lived inside ``pipelines/train.py`` and was reachable only from training,
which is why ``report`` could not follow ``infer``/``predict``: the hook was in
the wrong building, not missing. Nothing in it is training-specific -- it reads
``config.reporting`` and ``config.run.seed`` duck-typed and hands a run directory
to ``generate_report``, which is itself entirely artifact-driven.

It is moved here rather than imported across pipelines because
``pipelines/infer.py`` importing ``pipelines/train.py`` would make the inference
verb depend on the training verb for a capability that belongs to neither -- it
belongs to reporting. Both pipelines now call inward, which is the direction the
layering rule requires anyway.

``pipelines/train.py`` re-exports these under their former private names, so the
existing call sites and the source-text pins that guard this hook's behaviour
keep resolving to this definition (``inspect.getsource`` follows the object, not
the module it is read through).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Tables a run gets when its YAML declares no ``reporting:`` block. The floor:
#: no figures, no HTML, no optional deps — the CSV/MD/TeX summary of what the run
#: measured. See ``tables/run_summary.py`` for why the publication tables cannot
#: serve this role.
UNCONFIGURED_REPORT_TABLES = ["tab_run_summary"]


def run_unconfigured_report(run_dir: Path, logger_: Any) -> None:
    """Tables-only floor for a run that configured no reporting.

    Until now this path ran ``MetricsReportGenerator`` — unconditionally, with no
    config gate at all — while the canonical ``generate_report`` sat behind
    ``reporting.enabled``, which defaults False. So the *legacy* generator ran on
    every run and the *canonical* pipeline almost never did, which is the SSOT
    inverted. One entry point now, with a graduated default.

    Deliberately cheap: no figures, no HTML, no plotly. A run that wants figures
    declares ``reporting.enabled: true``.
    """
    try:
        from mriforge.infrastructure.reporting import generate_report
    except Exception as exc:
        logger_.warning("reporting floor: import failed (%s)", exc)
        return
    try:
        result = generate_report(
            run_dir,
            figures=[],
            tables_=UNCONFIGURED_REPORT_TABLES,
            qc_figures=False,
            html_report=False,
            interactive=False,
            emit_manifest=False,
        )
        written = [t for t, v in result.get("tables", {}).items() if v]
        if written:
            logger_.info(
                "report floor: %s → %s (declare `reporting.enabled: true` for figures)",
                ", ".join(written),
                result.get("out_dir"),
            )
    except Exception as exc:
        logger_.warning("reporting floor: generation failed (%s)", exc)


def maybe_run_reporting(config: Any, *, run_dir: Path, logger_: Any) -> None:
    """Run the reporting pipeline — full when configured, tables-only when not.

    ``generate_report`` is the single end-of-training report path. Imported
    lazily so a missing optional dep cannot break training.
    """
    reporting = getattr(config, "reporting", None)
    if reporting is None or not getattr(reporting, "enabled", False):
        run_unconfigured_report(run_dir, logger_)
        return
    try:
        from mriforge.infrastructure.reporting import generate_report
    except Exception as exc:
        logger_.warning("reporting hook: import failed (%s)", exc)
        return
    try:
        result = generate_report(
            run_dir,
            task=getattr(
                getattr(reporting, "task", "default"),
                "value",
                getattr(reporting, "task", "default"),
            ),
            style=getattr(
                getattr(reporting, "style", "nature"),
                "value",
                getattr(reporting, "style", "nature"),
            ),
            formats=tuple(getattr(reporting, "formats", ["pdf", "png"])),
            dpi=getattr(reporting, "dpi", 600),
            panel_labels=getattr(reporting, "panel_labels", True),
            method_name=getattr(reporting, "method_name", None) or run_dir.name,
            figures=getattr(reporting, "figures", None),
            tables_=getattr(reporting, "tables", None),
            metrics=getattr(reporting, "metrics", None),
            cohort=getattr(reporting, "cohort", None),
            hyperparameters=getattr(reporting, "hyperparameters", None),
            seed=config.run.seed,  # `run.seed` since phase 4b; the flat read stamped None
            extra_runs=getattr(reporting, "extra_runs", None),
            out_subdir=getattr(reporting, "out_subdir", "report"),
            emit_manifest=getattr(reporting, "emit_manifest", True),
            submission_bundle=getattr(reporting, "submission_bundle", False),
            tikz=getattr(reporting, "tikz", False),
            qc_figures=getattr(reporting, "qc_figures", True),
            html_report=getattr(reporting, "html_report", True),
            interactive=getattr(reporting, "interactive", True),
        )
        logger_.info(
            "reporting hook: %d figures + %d tables + %d tikz → %s",
            sum(1 for v in result.get("figures", {}).values() if v is not None),
            sum(1 for v in result.get("tables", {}).values() if v is not None),
            len(result.get("tikz", {})),
            result.get("out_dir"),
        )
    except Exception as exc:
        if getattr(reporting, "fail_on_error", False):
            raise
        logger_.warning("reporting hook: generation failed (%s)", exc)
