r"""Figure 1.9 — Ablation strip plot.

Horizontal bars of ΔMetric (with 95 % CI) when each component is removed.
Zero line emphasized. Directly answers "is each component necessary?"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from mriforge.infrastructure.reporting.plotters._stats import t_ci_half_width
from mriforge.infrastructure.reporting.style import save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    primary_metric: str = "psnr",
    full_method: str = "full",
    higher_is_better: bool = True,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the ablation strip.

    Args:
        primary_metric: Metric used for the Δ comparison.
        full_method: Method name representing the unablated full model.
            Other methods are interpreted as ablations of it.
        higher_is_better: Direction of "better" — controls bar colour.
    """
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns:
        return None
    sub = df[df["metric"] == primary_metric].copy()
    if sub.empty or full_method not in sub["method"].unique():
        return None

    # Per-method per-subject samples (or per-run mean if no subject_id)
    per_method = (
        sub.groupby(["method", "subject_id"])["value"].mean()
        if "subject_id" in sub.columns
        else sub.groupby("method")["value"].apply(list)
    )

    # Build mean + 95% CI per method
    means: dict[str, float] = {}
    cis: dict[str, tuple[float, float]] = {}
    for method, vals in (
        per_method.groupby("method") if hasattr(per_method, "groupby") else per_method.items()
    ):
        arr = np.asarray(list(vals)).astype(float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        m = arr.mean()
        # Student-t CI half-width (sample std, t critical value). The previous
        # z=1.96 factor is far too narrow at the seed counts here — see
        # _stats.t_ci_half_width.
        half = t_ci_half_width(arr)
        means[method] = m
        cis[method] = (m - half, m + half)

    full_mean = means.get(full_method)
    if full_mean is None:
        return None
    deltas = {m: v - full_mean for m, v in means.items() if m != full_method}
    if not deltas:
        return None

    use_default_style()
    fig, ax = plt.subplots(figsize=(3.6, max(2.0, 0.35 * len(deltas) + 1.0)))
    methods = sorted(deltas.keys(), key=lambda k: deltas[k])
    y = np.arange(len(methods))
    vals = np.asarray([deltas[m] for m in methods])
    err_lo = np.asarray([abs(deltas[m] - (cis[m][0] - full_mean)) for m in methods])
    err_hi = np.asarray([abs((cis[m][1] - full_mean) - deltas[m]) for m in methods])
    colours = ["#D55E00" if (v > 0) == higher_is_better else "#0072B2" for v in vals]
    ax.barh(
        y,
        vals,
        xerr=[err_lo, err_hi],
        color=colours,
        alpha=0.85,
        error_kw={"elinewidth": 0.8, "capsize": 2},
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=7)
    arrow = "↑" if higher_is_better else "↓"
    ax.set_xlabel(f"Δ {primary_metric}  ({arrow} better)")
    ax.set_title(f"Ablation vs. '{full_method}'", fontsize=8)

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
