r"""Figure 1.16 — Run summary card.

A one-glance triage card: run provenance facts (``split="run"`` rows from
``run_summary.json``) side by side with the best metrics achieved
(``split="best"`` rows from ``final_metrics.json``) and the final scalars.
Rendered as two matplotlib tables so the card obeys the house style and
lands in the same PDF/PNG pipeline as every other figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mriforge.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from mriforge.infrastructure.reporting.style import (
    column_width,
    pretty_label,
    save_figure,
    use_default_style,
)

# split="run" facts use the computational-profile formats; everything else
# falls back to general-purpose float formatting.
_FACT_FORMATS = {
    "params_m": "{:.2f} M",
    "iterations_per_sec": "{:.2f} it/s",
    "duration_min": "{:.1f} min",
    "effective_batch": "{:.0f}",
    "early_stopping_best_iteration": "{:.0f}",
}


def _fmt(metric: str, value: float) -> str:
    return _FACT_FORMATS.get(metric, "{:.4g}").format(value)


def _rows_for(df: pd.DataFrame, split: str) -> list[tuple[str, str]]:
    sub = df[df["split"] == split].dropna(subset=["value"])
    # keep last occurrence per metric (multi-run frames: last run wins)
    sub = sub.groupby("metric", sort=True)["value"].last()
    return [(pretty_label(str(m)), _fmt(str(m), float(v))) for m, v in sub.items()]


def _render_table(ax, title: str, rows: list[tuple[str, str]]) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left")
    if not rows:
        ax.text(0.0, 0.5, "no data", fontsize=6, color="#888888")
        return
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colWidths=[0.62, 0.38],
        loc="upper left",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.1)
    for (_r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        if c == 1:
            cell.get_text().set_ha("right")


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Render the run-summary card. Returns ``None`` when no summary rows exist."""
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns or "split" not in df.columns:
        return None
    run_rows = _rows_for(df, "run")
    best_rows = _rows_for(df, "best")
    final_rows = _rows_for(df, "final")
    if not (run_rows or best_rows or final_rows):
        return None

    use_default_style()
    fig, axes = plt.subplots(1, 2, figsize=(column_width("onehalf"), 2.2))
    _render_table(axes[0], "Run profile", run_rows + final_rows)
    _render_table(axes[1], "Best metrics", best_rows)
    method = str(df["method"].iloc[0]) if "method" in df.columns and len(df) else ""
    if method:
        fig.suptitle(method, x=0.02, ha="left")
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
