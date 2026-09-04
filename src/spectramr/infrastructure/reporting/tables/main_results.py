r"""Table 2.1 — Main results table.

Rows = methods, columns = metrics; mean ± std over seeds; best in
**bold**, second-best *underlined*; significance superscript (e.g. ``*``
for p < 0.05 vs. baseline after Holm correction). Includes a parameter /
FLOPs cost column where present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER
from spectramr.core.metrics.statistical_tests import StatisticalTests


def _holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction.

    Delegates to the metrics SSOT
    (:meth:`spectramr.core.metrics.statistical_tests.StatisticalTests.holm_bonferroni_correction`).
    Kept as a thin wrapper so call sites in this table don't need to
    know the SSOT API surface.
    """
    if not pvals:
        return []
    out = StatisticalTests.holm_bonferroni_correction(pvals, alpha=0.05)
    return list(out["corrected_p_values"])


def _format_value(mean: float, std: float | None, decimals: int) -> str:
    if std is None or np.isnan(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def make(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    metrics: list[str] | None = None,
    cost_metrics: list[str] = ["params_m"],
    higher_is_better: dict[str, bool] | None = None,
    baseline_method: str | None = None,
    decimals: int = 3,
) -> dict[str, Path] | None:
    """Build the main results table in Markdown + LaTeX.

    Args:
        metrics: List of metric names to include. Defaults to common test
            metrics in the test split.
        cost_metrics: Cost columns appended to the right (e.g. ``params_m``).
        higher_is_better: Per-metric direction of "better" (default True
            for everything except metrics whose name contains ``loss``,
            ``error``, ``rmse``, ``nmse``, ``mae``, ``fid``, ``kid``,
            ``hd95``, ``ece``, ``nll``).
        baseline_method: If supplied, run paired t-test of every other
            method against the baseline and Holm-correct.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return None
    test = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    if test.empty:
        # fall back to last value of every metric in val split
        test = df[df["split"] == "val"].copy() if "split" in df.columns else df.copy()

    if metrics is None:
        common = {
            "psnr",
            "ssim",
            "ms_ssim",
            "lpips",
            "vif",
            "fsim",
            "haarpsi",
            "rmse",
            "nmse",
            "nrmse",
            "hfen",
            "gmsd",
            "fid",
            "kid",
            "dice",
            "hd95",
            "auroc",
            "ece",
        }
        metrics = sorted(set(test["metric"].unique()) & common)
    if not metrics:
        return None

    higher_is_better = higher_is_better or {}

    def hib(m: str) -> bool:
        # 1. caller override wins
        if m in higher_is_better:
            return higher_is_better[m]
        ml = m.lower()
        # 2. metric-direction SSOT (exact, then longest-token match) — the same
        #    source the pipelines layer uses, so a lower-is-better metric whose
        #    name lacks a hardcoded keyword (cjv, efc, bland_altman_bias,
        #    ghosting_ratio, …) is no longer bolded as "best" when it's worst.
        if ml in METRIC_HIGHER_IS_BETTER:
            return METRIC_HIGHER_IS_BETTER[ml]
        matches = [(k, v) for k, v in METRIC_HIGHER_IS_BETTER.items() if k in ml]
        if matches:
            return max(matches, key=lambda kv: len(kv[0]))[1]
        # 3. last-resort heuristic for non-registry cost/aux columns
        lower_keywords = (
            "loss",
            "error",
            "rmse",
            "nmse",
            "mae",
            "fid",
            "kid",
            "hd95",
            "ece",
            "nll",
            "lpips",
            "hfen",
            "gmsd",
            "asd",
        )
        return not any(kw in ml for kw in lower_keywords)

    methods = sorted(test["method"].unique())
    rows: list[dict] = []
    for m in methods:
        row: dict = {"method": m}
        for metric in metrics + cost_metrics:
            vals = (
                test[(test["method"] == m) & (test["metric"] == metric)]["value"]
                .dropna()
                .to_numpy()
                .astype(float)
            )
            if vals.size == 0:
                row[metric] = (np.nan, np.nan, np.nan)
            else:
                row[metric] = (vals.mean(), vals.std(ddof=1) if vals.size > 1 else np.nan, vals)
        rows.append(row)

    # Find best and second-best per metric column
    best_idx: dict[str, int] = {}
    second_idx: dict[str, int] = {}
    for metric in metrics:
        means = [r[metric][0] for r in rows]
        valid = [(i, v) for i, v in enumerate(means) if not np.isnan(v)]
        if not valid:
            continue
        valid.sort(key=lambda iv: iv[1], reverse=hib(metric))
        best_idx[metric] = valid[0][0]
        if len(valid) > 1:
            second_idx[metric] = valid[1][0]

    # Significance against baseline (paired Wilcoxon → t fallback)
    sig: dict[tuple[str, str], bool] = {}
    if baseline_method is not None and baseline_method in methods:
        base = next(r for r in rows if r["method"] == baseline_method)
        for metric in metrics:
            base_arr = base[metric][2]
            if not isinstance(base_arr, np.ndarray) or base_arr.size < 2:
                continue
            pvals: list[float] = []
            others: list[str] = []
            for r in rows:
                if r["method"] == baseline_method:
                    continue
                arr = r[metric][2]
                if not isinstance(arr, np.ndarray) or arr.size < 2:
                    continue
                # Paired across the *minimum* common length
                k = min(arr.size, base_arr.size)
                try:
                    res = stats.ttest_rel(arr[:k], base_arr[:k])
                    pvals.append(float(res.pvalue))
                    others.append(r["method"])
                except Exception:
                    pvals.append(1.0)
                    others.append(r["method"])
            adj = _holm_correction(pvals)
            for i, m in enumerate(others):
                sig[(m, metric)] = adj[i] < 0.05

    # Render Markdown
    cols = ["Method"] + metrics + cost_metrics
    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in enumerate(rows):
        cells = [str(r["method"])]
        for metric in metrics:
            mean, std, _ = r[metric]
            text = _format_value(mean, std, decimals)
            if i == best_idx.get(metric):
                text = f"**{text}**"
            elif i == second_idx.get(metric):
                text = f"_{text}_"
            if sig.get((r["method"], metric)):
                text += "*"
            cells.append(text)
        for cm in cost_metrics:
            mean, _, _ = r[cm]
            cells.append(_format_value(mean, None, 1) if not np.isnan(mean) else "—")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_path = out_dir / "tab_2_1_main_results.md"
    md_path.write_text("\n".join(md_lines))

    # Render LaTeX (IEEE style)
    tex_lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Main results. Mean $\pm$ std over seeds; best in \textbf{bold},"
        r" second-best in \emph{italics}; * $p<0.05$ vs. baseline (Holm-corrected).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l" + "c" * (len(metrics) + len(cost_metrics)) + "}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        cells = [str(r["method"]).replace("_", r"\_")]
        for metric in metrics:
            mean, std, _ = r[metric]
            text = _format_value(mean, std, decimals)
            if i == best_idx.get(metric):
                text = r"\textbf{" + text + "}"
            elif i == second_idx.get(metric):
                text = r"\emph{" + text + "}"
            if sig.get((r["method"], metric)):
                text += r"$^{*}$"
            cells.append(text)
        for cm in cost_metrics:
            mean, _, _ = r[cm]
            cells.append(_format_value(mean, None, 1) if not np.isnan(mean) else "—")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_dir / "tab_2_1_main_results.tex"
    tex_path.write_text("\n".join(tex_lines))

    # Machine-readable CSV sink: raw mean/std/cost values (no markup) so
    # downstream analysis doesn't have to re-parse the formatted LaTeX.
    # Function-local import avoids the tables/__init__ circular import
    # (the package imports this submodule before emit_table_csv is defined).
    from spectramr.infrastructure.reporting.tables import emit_table_csv

    flat: list[dict] = []
    for r in rows:
        rec: dict = {"method": r["method"]}
        for metric in metrics:
            mean, std, _ = r[metric]
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = std
        for cm in cost_metrics:
            rec[cm] = r[cm][0]
        flat.append(rec)
    csv_path = emit_table_csv(pd.DataFrame(flat), out_dir, "tab_2_1_main_results")

    return {"markdown": md_path, "latex": tex_path, "csv": csv_path}
