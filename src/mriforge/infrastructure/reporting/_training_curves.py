r"""Shared helper for the auto-generated experiment-report figures
(``MetricsReportGenerator._plot_losses`` and friends).

Design rationale (2026-05-05 reporting refactor)
-------------------------------------------------
The previous ``MetricsReportGenerator`` used raw matplotlib with
2-pixel markers, no smoothing, a single y-axis for losses spanning
multiple orders of magnitude, and no shared style with the rest of
the reporting pipeline. The result was unreadable on dense training
runs.

This helper centralises:

* **EMA smoothing** (configurable half-life) overlaid on a faint raw
  curve so reviewers see both noise and trend.
* **Auto log-vs-linear y-axis** based on dynamic range.
* **Final-value annotation** at the right edge of each curve, plus
  a "best step / value" callout in the figure subtitle.
* **Consistent colours and fonts** matching the rest of the reporting
  pipeline (Okabe-Ito palette via :func:`use_default_style`).
* **Clean legend placement** outside the plot area so it never
  occludes the curves.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.style import (
    colour_for,
    use_default_style,
)

logger = logging.getLogger(__name__)

# Local style for training-curve figures (slightly larger than IEEE
# main since these are 12-inch wide screen / dashboard figures, not
# single-column publication panels).
_FS_TITLE = 13
_FS_AXIS = 11
_FS_LEGEND = 9
_FS_NOTE = 8
_FS_TICK = 9


# ──────────────────────────────────────────────────────────────────────
# Smoothing + axis helpers
# ──────────────────────────────────────────────────────────────────────


def ema(values: np.ndarray, half_life: float = 8.0) -> np.ndarray:
    """Exponential moving average with the given (in-step) half-life."""
    if values.size == 0:
        return values.copy()
    alpha = 1.0 - np.exp(-np.log(2.0) / max(half_life, 1.0))
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, values.size):
        if np.isnan(values[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _should_use_log(values: np.ndarray) -> bool:
    """Pick a log y-axis when finite values span ≥ 2 orders of magnitude
    AND are strictly positive."""
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size < 4:
        return False
    return float(finite.max() / max(finite.min(), 1e-12)) >= 100.0


# ──────────────────────────────────────────────────────────────────────
# The single shared training-curve plot
# ──────────────────────────────────────────────────────────────────────


def plot_training_curves(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    title: str,
    ylabel: str,
    save_path: Path,
    higher_is_better: bool | None = None,
    smoothing_half_life: float = 8.0,
    log_y: bool | None = None,
    figsize: tuple[float, float] = (12.0, 5.5),
    dpi: int = 110,
    column_legend_names: Mapping[str, str] | None = None,
) -> Path | None:
    """Render a multi-curve training plot with EMA smoothing.

    Args:
        df:              The metrics DataFrame (one row per logged step).
        columns:         Names of columns to plot; missing columns are skipped.
        title:           Figure title.
        ylabel:          Y-axis label.
        save_path:       Output PNG path.
        higher_is_better: If True/False, an arrow ``▲``/``▼`` is added to
                         the title and the per-curve final-value box
                         shows the best value over all steps.
        smoothing_half_life: EMA half-life in steps (default 8).
        log_y:           If None, auto-decide via :func:`_should_use_log`.
        figsize, dpi:    Matplotlib figure size / dpi.
        column_legend_names: Optional override label per column.

    Returns:
        ``save_path`` on success, else ``None``.
    """
    cols = [c for c in columns if c in df.columns]
    if not cols:
        logger.debug("plot_training_curves: no columns from %s in df", columns)
        return None

    use_default_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    legend_handles = []
    all_finite = []
    for i, col in enumerate(cols):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        all_finite.append(values[np.isfinite(values)])
        steps = df.index.to_numpy()
        smoothed = ema(values, half_life=smoothing_half_life)
        colour = colour_for(col, fallback_index=i + 1)
        legend_label = column_legend_names.get(col, col) if column_legend_names else col

        ax.plot(steps, values, color=colour, alpha=0.18, linewidth=0.8)
        (line,) = ax.plot(
            steps,
            smoothed,
            color=colour,
            linewidth=1.8,
            label=legend_label,
        )
        legend_handles.append(line)

        # Right-edge final-value annotation
        last_finite = next(
            (v for v in reversed(values) if np.isfinite(v)),
            None,
        )
        if last_finite is not None:
            ax.text(
                steps[-1] + 0.01 * (steps[-1] - steps[0] + 1),
                last_finite,
                f" {last_finite:.4g}",
                color=colour,
                va="center",
                fontsize=_FS_NOTE,
                fontweight="bold",
            )

    if not legend_handles:
        plt.close(fig)
        return None

    # Auto log/linear axis
    flat = np.concatenate(all_finite) if all_finite else np.array([0.0])
    if log_y is None:
        log_y = _should_use_log(flat)
    if log_y:
        # Shift floor to avoid zero on log scale
        positive_min = float(flat[flat > 0].min()) if (flat > 0).any() else 1e-9
        ax.set_yscale("log")
        ax.set_ylim(bottom=positive_min * 0.7)

    # Subtitle showing best step / value if polarity known
    suffix = ""
    if higher_is_better is not None:
        for col in cols:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(v).any():
                continue
            if higher_is_better:
                idx = int(np.nanargmax(v))
            else:
                idx = int(np.nanargmin(v))
            suffix = f"  (best {col}: {v[idx]:.4g} @ step {df.index[idx]})"
            break

    arrow = "" if higher_is_better is None else (" ▲" if higher_is_better else " ▼")
    ax.set_title(f"{title}{arrow}{suffix}", fontsize=_FS_TITLE, fontweight="bold")
    ax.set_xlabel("Step", fontsize=_FS_AXIS)
    ax.set_ylabel(ylabel, fontsize=_FS_AXIS)
    ax.tick_params(labelsize=_FS_TICK)
    ax.grid(True, alpha=0.25)
    # Legend OUTSIDE the axes so it never occludes curves
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        fontsize=_FS_LEGEND,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved training-curve figure: %s", save_path)
    return save_path


__all__ = ["ema", "plot_training_curves"]
