r"""Figure A.7 — k-space → image reconstruction.

Per-row layout (7 columns):

    zero-filled | CG-SENSE | proposed | fully-sampled |
    residual (proposed − FS) + colorbar | residual histogram | radial k-error

Each magnitude image annotated with PSNR / SSIM / L1 versus FS reference.
The radial k-error panel is on a log-y axis with bin-width-normalised
mean absolute k-space error per |k| bin.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from mriforge.infrastructure.reporting.plotters._comparison_helpers import (
    _format_metric_str,
    add_image_panel,
    add_residual_histogram,
    add_residual_panel,
)
from mriforge.infrastructure.reporting.style import use_default_style


def _radial_error(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fp = np.fft.fftshift(np.fft.fft2(pred))
    fr = np.fft.fftshift(np.fft.fft2(ref))
    err = np.abs(fp - fr)
    h, w = err.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(err.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    counts = np.bincount(r.ravel()).clip(min=1)
    binned = np.bincount(r.ravel(), err.ravel()) / counts
    return np.arange(len(binned)), binned


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the k-space reconstruction figure.

    Args:
        cases: dicts with arrays ``zero_filled``, ``baseline``, ``pred``,
            ``fs_ref`` (all magnitude images, same H×W).
    """
    if cases is None or len(cases) == 0:
        return None
    out_path = Path(out_path)

    use_default_style()
    n_rows = len(cases)
    n_cols = 7
    col_w, row_h = 1.6, 1.6
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * col_w, n_rows * row_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )
    titles = [
        "zero-filled",
        "CG-SENSE",
        "proposed",
        "fully-sampled",
        "residual (ours − FS)",
        "residual histogram",
        "radial k-error",
    ]

    for r, c in enumerate(cases):
        zf = np.asarray(c["zero_filled"]).squeeze().astype(float)
        bl = np.asarray(c["baseline"]).squeeze().astype(float)
        pr = np.asarray(c["pred"]).squeeze().astype(float)
        ref = np.asarray(c["fs_ref"]).squeeze().astype(float)
        row_label = c.get("name")

        add_image_panel(
            axes[r][0],
            zf,
            title=titles[0] if r == 0 else None,
            row_label=row_label,
            metric_str=_format_metric_str(zf, ref),
        )
        add_image_panel(
            axes[r][1],
            bl,
            title=titles[1] if r == 0 else None,
            metric_str=_format_metric_str(bl, ref),
        )
        add_image_panel(
            axes[r][2],
            pr,
            title=titles[2] if r == 0 else None,
            metric_str=_format_metric_str(pr, ref),
        )
        add_image_panel(
            axes[r][3],
            ref,
            title=titles[3] if r == 0 else None,
        )
        abs_max, residual = add_residual_panel(
            axes[r][4],
            pr,
            ref,
            title=titles[4] if r == 0 else None,
            fig=fig,
        )
        add_residual_histogram(
            axes[r][5],
            residual,
            abs_max=abs_max,
            title=titles[5] if r == 0 else None,
        )

        # 7. Radial k-error
        ax = axes[r][6]
        for img, label, colour in (
            (zf, "ZF", "#888"),
            (bl, "CG-SENSE", "#0072B2"),
            (pr, "ours", "#D55E00"),
        ):
            k, e = _radial_error(img, ref)
            ax.semilogy(k[1:], e[1:].clip(min=1e-12), color=colour, linewidth=0.9, label=label)
        if r == 0:
            ax.set_title(titles[6], fontsize=8)
            ax.legend(frameon=False, fontsize=6, loc="upper right")
        ax.set_xlabel("|k|", fontsize=7)
        ax.set_ylabel("|Δ k|", fontsize=7)
        ax.tick_params(labelsize=6)

    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    if metadata is not None:
        write_sidecar(out_path, metadata)
    return out_path
