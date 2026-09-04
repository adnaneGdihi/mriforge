r"""Metric vs acceleration factor R (with ±1σ seed bands)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric=None,
    accel_col="acceleration",
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if df is None or df.empty or accel_col not in df.columns or "metric" not in df.columns:
        return None
    if metric is None:
        metric = "psnr" if "psnr" in set(df["metric"]) else df["metric"].iloc[0]
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    use_default_style("nature")
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single") * 0.8))
    for i, method in enumerate(sub["method"].unique()):
        md = sub[sub["method"] == method]
        g = md.groupby(accel_col)["value"]
        # Sample std (ddof=1) for the cross-seed +/-1 sigma band; see learning_curves.
        mean, std = g.mean(), g.std(ddof=1).fillna(0)
        ax.plot(mean.index, mean.values, marker="o", color=colour_for(method, i), label=method)
        ax.fill_between(
            mean.index, mean - std, mean + std, color=colour_for(method, i), alpha=0.2, linewidth=0
        )
    ax.set_xlabel("acceleration factor R")
    ax.set_ylabel(metric.upper())
    ax.legend(loc="best", fontsize=5)
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
