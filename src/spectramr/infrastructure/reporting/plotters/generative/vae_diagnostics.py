r"""VAE diagnostics: reconstruction term vs. KL term over training,
posterior collapse check (per-latent KL distribution).
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
    has_rec = "reconstruction_loss" in set(df["metric"].unique())
    has_kl = "kl_loss" in set(df["metric"].unique()) or "kl" in set(df["metric"].unique())
    has_per_latent = any("kl_latent_" in m for m in df["metric"].unique())
    if not (has_rec or has_kl or has_per_latent):
        return None

    use_default_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))

    # Left: rec vs KL over training
    ax_l = axes[0]
    plotted = False
    if has_rec:
        rec = df[df["metric"] == "reconstruction_loss"].groupby("step")["value"].mean()
        ax_l.plot(rec.index, rec.values, color="#0072B2", label="reconstruction")
        plotted = True
    if has_kl:
        kl_metric = "kl_loss" if "kl_loss" in set(df["metric"].unique()) else "kl"
        kl = df[df["metric"] == kl_metric].groupby("step")["value"].mean()
        ax_l.plot(kl.index, kl.values, color="#D55E00", label="KL")
        plotted = True
    if plotted:
        ax_l.set_xlabel("step")
        ax_l.set_ylabel("loss")
        ax_l.set_yscale("log")
        ax_l.legend(frameon=False, fontsize=6)
        ax_l.set_title("(a) recon vs KL", fontsize=8)
    else:
        ax_l.set_axis_off()

    # Right: per-latent KL distribution (collapse check)
    ax_r = axes[1]
    if has_per_latent:
        cols = sorted(m for m in df["metric"].unique() if "kl_latent_" in m)
        # take last value per latent
        last = df[df["metric"].isin(cols)].sort_values("step").groupby("metric")["value"].last()
        ax_r.bar(range(len(last)), last.values, color="#009E73", alpha=0.85)
        ax_r.axhline(
            0.05,
            color="#888",
            linestyle="--",
            linewidth=0.6,
            label="collapse threshold (0.05 nats)",
        )
        ax_r.set_xlabel("latent dim")
        ax_r.set_ylabel("KL (nats)")
        ax_r.set_title("(b) per-latent KL", fontsize=8)
        ax_r.legend(frameon=False, fontsize=6)
    else:
        ax_r.set_axis_off()

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
