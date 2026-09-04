r"""Figure 1.5 — Predicted-vs-true scatter with marginals + fit stats."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    predictions_df: pd.DataFrame | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Joint plot with identity line, hexbin density, and marginals."""
    if predictions_df is None or predictions_df.empty:
        return None
    if not {"predicted", "target"}.issubset(predictions_df.columns):
        return None

    out_path = Path(out_path)
    pred = predictions_df["predicted"].to_numpy()
    targ = predictions_df["target"].to_numpy()

    use_default_style()
    fig = plt.figure(figsize=(4.0, 4.0))
    gs = fig.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:, :3])
    ax_x = fig.add_subplot(gs[0, :3], sharex=ax)
    ax_y = fig.add_subplot(gs[1:, 3], sharey=ax)

    # Hexbin density
    ax.hexbin(targ, pred, gridsize=40, cmap="viridis", mincnt=1)
    lo = float(min(targ.min(), pred.min()))
    hi = float(max(targ.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="white", linewidth=0.8)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")

    # Marginals
    ax_x.hist(targ, bins=40, color="#888", alpha=0.6)
    ax_y.hist(pred, bins=40, orientation="horizontal", color="#888", alpha=0.6)
    ax_x.tick_params(axis="x", labelbottom=False)
    ax_y.tick_params(axis="y", labelleft=False)

    # Stats
    rmse = float(np.sqrt(np.mean((pred - targ) ** 2)))
    if targ.std() > 0:
        r2 = 1.0 - float(np.var(pred - targ) / np.var(targ))
    else:
        r2 = float("nan")
    ax.text(
        0.04,
        0.95,
        f"$R^2$={r2:.3f}\nRMSE={rmse:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        fontsize=7,
    )

    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
