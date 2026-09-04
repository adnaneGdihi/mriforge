r"""Figure 1.2 — Learning curves with seed bands.

Train + validation loss vs. step on log-y, shaded ±1 σ band over seeds,
small-multiples grid (one panel per metric) instead of dual-axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metrics: list[str] | None = None,
    log_y: bool = True,
    band: bool = True,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the learning curves figure.

    Args:
        metrics: Metric names to plot (one panel each). Defaults to
            ``["loss", "val_loss"]`` if available, else any metric whose
            name contains "loss".
        log_y: Whether to use log-scale y axis.
        band: If True, shade ±1 σ across runs sharing the same ``method``.
    """
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns:
        return None
    if metrics is None:
        # heuristic — include loss-like metrics
        candidates = [m for m in df["metric"].unique() if "loss" in m.lower()]
        metrics = candidates[:6] or list(df["metric"].unique()[:4])
    sub = df[df["metric"].isin(metrics)].copy()
    if sub.empty:
        return None
    sub = sub[sub["step"].fillna(-1) >= 0]
    if sub.empty:
        return None

    use_default_style()
    n = len(metrics)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.4 * cols, 1.9 * rows), sharex=False, squeeze=False
    )
    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        per_metric = sub[sub["metric"] == metric]
        if per_metric.empty:
            ax.set_axis_off()
            continue
        for i, method in enumerate(per_metric["method"].unique()):
            method_df = per_metric[per_metric["method"] == method]
            grouped = method_df.groupby("step")["value"]
            mean = grouped.mean()
            # Sample std (ddof=1): the band is "+/-1 sigma over seeds", and the
            # seeds are a sample, not the population. ddof=0 understates sigma by
            # sqrt((n-1)/n) -- 29% too narrow at n=2, 18% at n=3 -- exactly the
            # seed counts used here. Single-seed groups -> NaN -> 0 (no band).
            std = grouped.std(ddof=1).fillna(0)
            colour = colour_for(str(method), fallback_index=i)
            ax.plot(mean.index, mean.values, color=colour, label=str(method))
            if band and not (std == 0).all():
                ax.fill_between(
                    mean.index, mean - std, mean + std, color=colour, alpha=0.2, linewidth=0
                )
        ax.set_title(pretty_label(metric))
        # log-scale only when the panel has positive data — matplotlib emits a
        # UserWarning (not an exception) on an all-zero/negative log axis, and
        # warnings are not OK (pitfall #10)
        if log_y and (per_metric["value"].dropna() > 0).any():
            ax.set_yscale("log")
        # x-label only on the bottom row (less ink, same information)
        if idx // cols == rows - 1 or idx + cols >= n:
            ax.set_xlabel(pretty_label("step"))
        if n > 1:
            panel_label(ax, chr(ord("a") + idx))
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc="best")

    # Trim empty axes
    for empty in range(n, rows * cols):
        axes[empty // cols][empty % cols].set_axis_off()

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
