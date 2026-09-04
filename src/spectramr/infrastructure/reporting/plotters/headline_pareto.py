r"""Figure 1.1 — Headline Pareto.

A 2-D scatter of *primary metric* vs. *cost* (params / FLOPs / runtime)
with one marker per method and the non-dominated frontier highlighted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    pretty_label,
    save_figure,
    use_default_style,
)


def _pareto_front(points: np.ndarray, *, maximise_y: bool = True) -> np.ndarray:
    """Return a boolean mask selecting non-dominated points.

    points: ``(n, 2)`` array of ``(cost, metric)``.
    """
    order = np.argsort(points[:, 0])  # ascending cost
    sorted_pts = points[order]
    is_front_sorted = np.zeros(len(points), dtype=bool)
    best = -np.inf if maximise_y else np.inf
    for i, p in enumerate(sorted_pts):
        if (maximise_y and p[1] > best) or (not maximise_y and p[1] < best):
            is_front_sorted[i] = True
            best = p[1]
    mask = np.zeros(len(points), dtype=bool)
    mask[order] = is_front_sorted
    return mask


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metric: str = "psnr",
    cost: str = "params_m",
    maximise_metric: bool = True,
    title: str | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the headline figure.

    Args:
        df: Long-format DataFrame from `aggregator.aggregate*`.
        out_path: Where to write the PDF/PNG.
        metric: Which row in the ``metric`` column to use as the y-axis.
        cost: Which row to use as x-axis (e.g. ``params_m``, ``flops_g``).
        maximise_metric: If True, "better" is up (PSNR, SSIM); if False, down (FID).
    """
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns:
        return None
    needed = {metric, cost}
    sub = df[df["metric"].isin(needed)].copy()
    if sub.empty or sub["method"].nunique() == 0:
        return None
    # Both axes must be present as metric rows; otherwise the pivot lacks the
    # column and ``dropna(subset=[metric, cost])`` would raise rather than
    # gracefully returning "no data" (e.g. cost column was never logged).
    present = set(sub["metric"].unique())
    if metric not in present or cost not in present:
        return None
    pivot = sub.pivot_table(
        index="method", columns="metric", values="value", aggfunc="last"
    ).dropna(subset=[metric, cost])
    if pivot.empty:
        return None

    use_default_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    pts = pivot[[cost, metric]].to_numpy()
    front_mask = _pareto_front(pts, maximise_y=maximise_metric)

    for i, (m, row) in enumerate(pivot.iterrows()):
        c = colour_for(str(m), fallback_index=i)
        ax.scatter(row[cost], row[metric], s=40, color=c, label=str(m), zorder=3)
        ax.annotate(
            str(m),
            (row[cost], row[metric]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            color=c,
        )

    # Front line
    if front_mask.sum() >= 2:
        front_pts = pts[front_mask]
        order = np.argsort(front_pts[:, 0])
        ax.plot(
            front_pts[order, 0],
            front_pts[order, 1],
            "--",
            color="#888888",
            linewidth=0.8,
            zorder=2,
            label="Pareto frontier",
        )

    ax.set_xlabel(pretty_label(cost))
    ax.set_ylabel(pretty_label(metric))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=6, loc="best")

    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
