r"""Headline qualitative reconstruction panel.

Rows = cases (best/median/worst by primary metric); columns =
[input, prediction, target, error]. Each row carries a magnified ROI inset
on the prediction and per-panel SSIM/PSNR annotations. A shared diverging
colorbar encodes the |prediction − target| error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    column_width,
    panel_label,
    save_figure,
    use_default_style,
)


def _norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    use_default_style("nature")
    n = len(cases)
    col_titles = ["Input", "Prediction", "Target", "Error ×"]
    fig, axes = plt.subplots(n, 4, figsize=(column_width("double"), 1.1 * n + 0.4), squeeze=False)
    panel_i = 0
    for r, case in enumerate(cases):
        inp = _norm(case.get("input", case["prediction"]))
        pred = _norm(case["prediction"])
        tgt = _norm(case["target"])
        err = np.abs(case["prediction"].astype(float) - case["target"].astype(float))
        err_n = err / (err.max() + 1e-8)
        panels = [inp, pred, tgt, err_n]
        cmaps = ["gray", "gray", "gray", "magma"]
        for c, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[r][c]
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0:
                ax.set_ylabel(case.get("rank", f"case {r}"))
            panel_label(ax, chr(ord("a") + panel_i))
            panel_i += 1
        # ROI inset (centre quarter) on the prediction column
        ax_pred = axes[r][1]
        h, w = pred.shape
        roi = pred[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        axin = ax_pred.inset_axes([0.62, 0.62, 0.36, 0.36])
        axin.imshow(roi, cmap="gray", vmin=0, vmax=1)
        axin.set_xticks([])
        axin.set_yticks([])
        for sp in axin.spines.values():
            sp.set_edgecolor("#E69F00")
            sp.set_linewidth(0.8)
        # metric annotation on the error column
        m = case.get("metrics", {})
        txt = "  ".join(f"{k.upper()} {v:.3g}" for k, v in m.items() if k in ("ssim", "psnr"))
        if txt:
            axes[r][3].set_xlabel(txt, fontsize=5)
    fig.colorbar(im, ax=axes[:, 3].tolist(), shrink=0.6, label="|pred − target|")
    # explicit margins instead of tight_layout: the shared colorbar spanning the
    # error column is incompatible with tight_layout (emits a UserWarning).
    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.08, wspace=0.05, hspace=0.2)
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
