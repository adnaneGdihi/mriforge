r"""Table 2.3 — Hyperparameter table.

Searched range, distribution (log-uniform, etc.), final value, and
selection criterion. Written as Markdown + LaTeX from a structured dict.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def make(
    df,
    out_dir: str | Path,
    *,
    hyperparameters: list[dict] | None = None,
) -> dict[str, Path] | None:
    """Render the hyperparameter table.

    Args:
        hyperparameters: list of dicts with keys
            ``{name, distribution, range, final, criterion}``.
    """
    if not hyperparameters:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "| Hyperparameter | Distribution | Range | Final | Selection criterion |",
        "|---|---|---|---|---|",
    ]
    tex_rows: list[str] = []
    for hp in hyperparameters:
        md_lines.append(
            f"| {hp.get('name', '?')} | {hp.get('distribution', '?')} | "
            f"{hp.get('range', '?')} | {hp.get('final', '?')} | "
            f"{hp.get('criterion', '?')} |"
        )
        tex_rows.append(
            f"{hp.get('name', '?').replace('_', '\\_')} & "
            f"{hp.get('distribution', '?')} & "
            f"{hp.get('range', '?')} & {hp.get('final', '?')} & "
            f"{hp.get('criterion', '?')} \\\\"
        )

    md_path = out_dir / "tab_2_3_hyperparameters.md"
    md_path.write_text("\n".join(md_lines))

    tex_lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Searched hyperparameters.}",
        r"\label{tab:hyperparameters}",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Hyperparameter & Distribution & Range & Final & Criterion \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = out_dir / "tab_2_3_hyperparameters.tex"
    tex_path.write_text("\n".join(tex_lines))

    # Machine-readable CSV sink. The hyperparameter list is already
    # structured, so the raw DataFrame is a direct serialisation.
    # Function-local import avoids the tables/__init__ circular import.
    from spectramr.infrastructure.reporting.tables import emit_table_csv

    csv_path = emit_table_csv(pd.DataFrame(hyperparameters), out_dir, "tab_2_3_hyperparameters")
    return {"markdown": md_path, "latex": tex_path, "csv": csv_path}
