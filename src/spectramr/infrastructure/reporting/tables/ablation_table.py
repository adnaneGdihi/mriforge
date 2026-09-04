r"""Table 2.2 — Ablation table.

Each row removes / replaces one component; final row is the full model.
``Δ`` from the full model in a dedicated column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    primary_metric: str = "psnr",
    full_method: str = "full",
    higher_is_better: bool = True,
    decimals: int = 3,
) -> dict[str, Path] | None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "metric" not in df.columns:
        return None
    sub = df[df["metric"] == primary_metric].copy()
    if sub.empty or full_method not in sub["method"].unique():
        return None

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for m in sub["method"].unique():
        v = sub[sub["method"] == m]["value"].dropna().to_numpy().astype(float)
        if v.size == 0:
            continue
        means[m] = float(v.mean())
        stds[m] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    full_mean = means[full_method]
    arrow = "↑" if higher_is_better else "↓"

    md_lines = [
        f"| Variant | {primary_metric} | Δ (vs full, {arrow} better) |",
        "|---|---|---|",
    ]
    tex_rows: list[str] = []
    # sort: largest |Δ| first, full last
    ordered = sorted(
        (m for m in means if m != full_method),
        key=lambda m: -abs(means[m] - full_mean),
    ) + [full_method]
    for m in ordered:
        mean = means[m]
        std = stds.get(m)
        delta = mean - full_mean if m != full_method else 0.0
        s = f"{mean:.{decimals}f}" + (f" ± {std:.{decimals}f}" if not np.isnan(std) else "")
        d = f"{delta:+.{decimals}f}" if m != full_method else "—"
        md_lines.append(f"| {m} | {s} | {d} |")
        tex_rows.append(f"{m.replace('_', r'\\_')} & {s} & {d} \\\\")

    md_path = out_dir / "tab_2_2_ablation.md"
    md_path.write_text("\n".join(md_lines))

    tex_lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Ablation. Mean $\pm$ std on " + primary_metric + ".}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"Variant & {primary_metric} & $\\Delta$ \\\\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = out_dir / "tab_2_2_ablation.tex"
    tex_path.write_text("\n".join(tex_lines))

    # Machine-readable CSV sink (raw values, no markup). Function-local
    # import avoids the tables/__init__ circular import.
    from spectramr.infrastructure.reporting.tables import emit_table_csv

    flat = [
        {
            "variant": m,
            primary_metric: means[m],
            f"{primary_metric}_std": stds.get(m, float("nan")),
            "delta_vs_full": (means[m] - full_mean) if m != full_method else 0.0,
        }
        for m in ordered
    ]
    csv_path = emit_table_csv(pd.DataFrame(flat), out_dir, "tab_2_2_ablation")
    return {"markdown": md_path, "latex": tex_path, "csv": csv_path}
