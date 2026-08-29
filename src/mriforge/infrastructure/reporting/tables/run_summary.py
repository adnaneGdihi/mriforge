r"""Table — per-run metric summary. The tables-only floor.

Every other table here is a *publication* table: ``tab_2_1_main_results`` compares
methods across seeds, ``tab_2_4_dataset_descriptor`` describes a cohort. Both
return ``None`` on a plain training run, because a single run has no second
method and no cohort descriptor. That is correct behaviour for those tables and
it left the default report with nothing to emit — the two-table ``default``
preset produced zero files on a run dir that had 27 rows of perfectly good
metrics sitting in ``logs/``.

This table consumes exactly what the aggregator already recovers from a training
run (``training_metrics.csv``, ``validation_metrics.csv``, ``final_metrics.json``)
and answers the question a researcher actually opens the run directory to ask:
*what did each metric end at, what was its best, and when?*

It is deliberately the cheapest possible artifact — no figures, no optional
dependencies, no cross-run joins — so it can be the guaranteed floor for a run
whose YAML declares no ``reporting:`` block at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mriforge.core.metrics.metric_directions import resolve_direction

#: Columns the aggregator guarantees on its long-format frame.
_REQUIRED = {"metric", "value", "split"}


def make(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    decimals: int = 4,
    **_ignored: object,
) -> dict[str, Path] | None:
    """Summarise every metric the run recorded, per split.

    One row per ``(split, metric)``: final value, best value, the step the best
    occurred at, and the observation count.

    ``direction`` comes from the metric-direction SSOT
    (:func:`~mriforge.core.metrics.metric_directions.resolve_direction`) — the
    NON-fatal resolver, which returns ``None`` for an unrecognised key rather
    than raising. A summary table must never be the thing that fails a run's
    wrap-up over an unfamiliar column name, and an honest empty cell is better
    than a guessed direction. Guessing is what made ``best_metric_name: lpips``
    maximise LPIPS.

    Returns ``None`` when the frame carries no usable rows, so an empty run
    emits no file rather than a header over nothing.
    """
    if df is None or df.empty or not _REQUIRED.issubset(df.columns):
        return None

    work = df.dropna(subset=["value"]).copy()
    if work.empty:
        return None

    # `step` is present on aggregator output but absent on hand-built frames.
    if "step" not in work.columns:
        work["step"] = pd.NA

    rows: list[dict[str, object]] = []
    for (split, metric), grp in work.groupby(["split", "metric"], dropna=False):
        grp = grp.sort_values("step", na_position="first")
        higher = resolve_direction(str(metric))
        # `idxmax`/`idxmin` need a direction; with none, "best" is not defined
        # and the column stays empty rather than defaulting to max (#208).
        if higher is None:
            best_value = best_step = None
        else:
            pick = grp["value"].idxmax() if higher else grp["value"].idxmin()
            best_value = grp.loc[pick, "value"]
            best_step = grp.loc[pick, "step"]

        rows.append(
            {
                "split": split,
                "metric": metric,
                "final": round(float(grp["value"].iloc[-1]), decimals),
                "best": (None if best_value is None else round(float(best_value), decimals)),
                "best_step": best_step,
                "n": len(grp),
                "direction": {True: "higher", False: "lower", None: ""}[higher],
            }
        )

    if not rows:
        return None

    # Deferred: `triple_emit` lives in this package's __init__, which imports
    # this module — a module-level import would be circular.
    from . import triple_emit

    table = pd.DataFrame(rows).sort_values(["split", "metric"]).reset_index(drop=True)
    return triple_emit(
        table,
        out_dir,
        "run_summary",
        caption="Per-metric run summary: final and best value with the step attained.",
        label="tab:run_summary",
        floatfmt=f".{decimals}f",
    )
