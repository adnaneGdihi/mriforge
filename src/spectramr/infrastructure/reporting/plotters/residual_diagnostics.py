r"""Figure 1.4 — Residual diagnostics 4-panel.

(a) residuals vs fitted, (b) Q–Q plot, (c) histogram with normal fit,
(d) residual ACF for sequential data. The standard regression-defense
panel — absence invites reviewer skepticism.

Inputs come from a *predictions DataFrame* shaped like
``columns=[predicted, target]``, supplied via ``predictions_df``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import save_figure, use_default_style


def _normal_qq(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_r = np.sort(residuals)
    n = len(sorted_r)
    quantiles = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    from scipy.stats import norm  # local — scipy already a dep

    theoretical = norm.ppf(quantiles)
    return theoretical, sorted_r


def _acf(residuals: np.ndarray, n_lags: int = 30) -> np.ndarray:
    x = residuals - residuals.mean()
    var = (x * x).mean()
    if var == 0:
        return np.zeros(n_lags)
    return np.array([((x[:-k] * x[k:]).mean() / var) if k > 0 else 1.0 for k in range(n_lags)])


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    predictions_df: pd.DataFrame | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the 2×2 residual diagnostics panel.

    Args:
        predictions_df: Long table with ``predicted`` and ``target``
            columns (per-subject rows). The aggregator can build this
            from a final-eval JSON containing per-subject predictions.
    """
    if predictions_df is None or predictions_df.empty:
        return None
    if not {"predicted", "target"}.issubset(predictions_df.columns):
        return None

    out_path = Path(out_path)
    pred = predictions_df["predicted"].to_numpy()
    targ = predictions_df["target"].to_numpy()
    residuals = pred - targ

    use_default_style()
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.0))
    (ax_rf, ax_qq), (ax_hi, ax_ac) = axes

    # (a) residuals vs fitted
    ax_rf.scatter(pred, residuals, s=8, alpha=0.6)
    ax_rf.axhline(0, color="#888", linewidth=0.6)
    ax_rf.set_xlabel("fitted")
    ax_rf.set_ylabel("residual")
    ax_rf.set_title("(a) residuals vs fitted", fontsize=8)

    # (b) Q-Q plot
    th, sr = _normal_qq(residuals)
    ax_qq.scatter(th, sr, s=8, alpha=0.6)
    lim = [min(th.min(), sr.min()), max(th.max(), sr.max())]
    ax_qq.plot(lim, lim, "--", color="#888", linewidth=0.6)
    ax_qq.set_xlabel("theoretical (N(0,1))")
    ax_qq.set_ylabel("sample")
    ax_qq.set_title("(b) Q-Q plot", fontsize=8)

    # (c) histogram + normal fit
    ax_hi.hist(residuals, bins=40, density=True, alpha=0.7, color="#56B4E9")
    mu, sig = residuals.mean(), residuals.std()
    if sig > 0:
        x = np.linspace(residuals.min(), residuals.max(), 200)
        ax_hi.plot(
            x,
            np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
            color="black",
            linewidth=0.8,
        )
    ax_hi.set_xlabel("residual")
    ax_hi.set_ylabel("density")
    ax_hi.set_title(f"(c) histogram (μ={mu:.2g}, σ={sig:.2g})", fontsize=8)

    # (d) ACF
    acf = _acf(residuals, n_lags=min(30, len(residuals) // 4))
    ax_ac.bar(np.arange(len(acf)), acf, width=0.7, color="#0072B2")
    ax_ac.axhline(0, color="#888", linewidth=0.6)
    if len(residuals) > 0:
        ci95 = 1.96 / np.sqrt(len(residuals))
        ax_ac.axhline(ci95, linestyle=":", color="#888", linewidth=0.6)
        ax_ac.axhline(-ci95, linestyle=":", color="#888", linewidth=0.6)
    ax_ac.set_xlabel("lag")
    ax_ac.set_ylabel("ACF")
    ax_ac.set_title("(d) residual ACF", fontsize=8)

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
