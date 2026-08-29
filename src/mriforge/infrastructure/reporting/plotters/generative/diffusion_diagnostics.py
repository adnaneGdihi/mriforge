r"""Diffusion / score-based diagnostics: noise-pred loss vs. timestep,
sample-quality vs. NFE, CFG-scale sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mriforge.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from mriforge.infrastructure.reporting.style import save_figure, use_default_style


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    if df.empty:
        return None
    out_path = Path(out_path)

    metric_per_t = "loss_per_timestep"  # convention for binned loss-vs-t
    quality_vs_nfe = "psnr_vs_nfe"
    cfg_sweep = "psnr_vs_cfg"

    rows: list[tuple[str, str, str]] = []
    for m, ylabel, xlabel in (
        (metric_per_t, "noise-pred loss", "timestep t"),
        (quality_vs_nfe, "PSNR", "NFE"),
        (cfg_sweep, "PSNR", "CFG scale"),
    ):
        if m in set(df["metric"].unique()):
            rows.append((m, ylabel, xlabel))
    if not rows:
        return None

    use_default_style()
    fig, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 2.0), squeeze=False)
    for i, (m, ylab, xlab) in enumerate(rows):
        ax = axes[0][i]
        sub = df[df["metric"] == m]
        # Use 'step' column as the abscissa (binned timestep / NFE / scale)
        med = sub.groupby("step")["value"].median()
        if not med.empty:
            ax.plot(med.index, med.values, color="#0072B2")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(m, fontsize=8)
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
