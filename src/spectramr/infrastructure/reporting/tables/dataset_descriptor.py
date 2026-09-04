r"""Table 2.4 — Dataset descriptor.

n, dimensionality, class balance, train/val/test split sizes, source DOI.

Emits .tex + .md + .csv via the canonical :func:`triple_emit` helper
(Phase 3.1 of `backlog_beautify_logging_and_reporting.md`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make(
    df,
    out_dir: str | Path,
    *,
    cohort: dict | None = None,
) -> dict[str, Path] | None:
    if not cohort:
        return None

    fields = [
        ("n_total", "Total subjects"),
        ("train_n", "Train"),
        ("val_n", "Validation"),
        ("test_n", "Test"),
        ("age_mean", "Age mean"),
        ("age_std", "Age std"),
        ("sex_split", "Sex split"),
        ("scanners", "Scanners"),
        ("field_strength", "Field strength"),
        ("pathology", "Pathology distribution"),
        ("split_rule", "Split rule"),
        ("doi", "Source / DOI"),
    ]
    descriptor_df = pd.DataFrame(
        [{"Field": label, "Value": cohort.get(k, "—")} for k, label in fields]
    )

    # Local import avoids circular import via package __init__.
    from spectramr.infrastructure.reporting.tables import triple_emit

    paths = triple_emit(
        descriptor_df,
        out_dir,
        stem="tab_2_4_dataset_descriptor",
        caption="Dataset descriptor. Splits are subject-level (no slice leakage).",
        label="tab:dataset",
    )
    return {"markdown": paths["md"], "latex": paths["tex"], "csv": paths["csv"]}
