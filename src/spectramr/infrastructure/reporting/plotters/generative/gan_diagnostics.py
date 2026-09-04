r"""GAN-specific diagnostics:
discriminator accuracy ≈ 0.5 at equilibrium, gradient-penalty norm,
generator/discriminator loss ratio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import save_figure, use_default_style


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
    keys = {
        "d_acc": "D acc",
        "discriminator_acc": "D acc",
        "g_loss": "G loss",
        "d_loss": "D loss",
        "gp_norm": "‖∇D‖",
        "gradient_penalty": "‖∇D‖",
    }
    avail = {short: lab for short, lab in keys.items() if short in set(df["metric"].unique())}
    if not avail:
        return None

    use_default_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
    # Left: D acc
    ax_acc = axes[0]
    plotted_acc = False
    for k in ("d_acc", "discriminator_acc"):
        if k in avail:
            d = df[df["metric"] == k].groupby("step")["value"].mean()
            ax_acc.plot(d.index, d.values, color="#0072B2", label=avail[k])
            plotted_acc = True
    if plotted_acc:
        ax_acc.axhline(0.5, color="#888", linestyle="--", linewidth=0.6)
        ax_acc.set_xlabel("step")
        ax_acc.set_ylabel("discriminator accuracy")
        ax_acc.set_title("(a) D-acc → 0.5 at equilibrium", fontsize=8)
        ax_acc.legend(frameon=False, fontsize=6)
    else:
        ax_acc.set_axis_off()

    # Right: G/D loss ratio
    ax_lr = axes[1]
    if "g_loss" in avail and "d_loss" in avail:
        g = df[df["metric"] == "g_loss"].groupby("step")["value"].mean()
        d = df[df["metric"] == "d_loss"].groupby("step")["value"].mean()
        common = g.index.intersection(d.index)
        ratio = (g.loc[common] / d.loc[common].replace(0, float("nan"))).dropna()
        ax_lr.plot(ratio.index, ratio.values, color="#D55E00")
        ax_lr.axhline(1.0, color="#888", linestyle="--", linewidth=0.6)
        ax_lr.set_xlabel("step")
        ax_lr.set_ylabel("G/D loss ratio")
        ax_lr.set_title("(b) loss ratio", fontsize=8)
    elif "gp_norm" in avail or "gradient_penalty" in avail:
        for k in ("gp_norm", "gradient_penalty"):
            if k in avail:
                g = df[df["metric"] == k].groupby("step")["value"].mean()
                ax_lr.plot(g.index, g.values, color="#009E73")
                ax_lr.axhline(1.0, color="#888", linestyle="--", linewidth=0.6)
        ax_lr.set_xlabel("step")
        ax_lr.set_ylabel("‖∇D‖")
        ax_lr.set_title("(b) gradient-penalty norm", fontsize=8)
    else:
        ax_lr.set_axis_off()

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
