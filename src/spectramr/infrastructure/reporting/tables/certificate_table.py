"""Per-certificate summary table (LaTeX + Markdown).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §V.3.

Builds a one-row-per-certificate table with columns:
    letter | certified | bound_value | target | n_calibration | passed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _row(
    letter: str,
    certified: bool | None,
    bound: float | None,
    target: float | None,
    n: int | None,
    passed: bool | None,
) -> dict[str, Any]:
    return {
        "letter": letter,
        "certified": "✓" if certified else ("✗" if certified is False else "—"),
        "bound_value": f"{bound:.4f}" if bound is not None else "—",
        "target": f"{target:.4f}" if target is not None else "—",
        "n_calibration": str(n) if n is not None else "—",
        "passed": "✓" if passed else ("✗" if passed is False else "—"),
    }


def build_certificate_rows(badge_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract one row per upstream certificate from a ValidationBadge JSON."""
    upstream = badge_payload.get("upstream_certificates", {})
    rows: list[dict[str, Any]] = []
    for letter, data in upstream.items():
        rows.append(
            _row(
                letter=letter,
                certified=data.get("certified"),
                bound=data.get("bound_value"),
                target=data.get("target"),
                n=data.get("n_calibration"),
                passed=data.get("passed"),
            )
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "| — | — | — | — | — | — |"
    cols = ["letter", "certified", "bound_value", "target", "n_calibration", "passed"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def render_latex(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return r"\begin{tabular}{c}--\end{tabular}"
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Letter & Certified & Bound & Target & $n_{\mathrm{cal}}$ & Passed \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['letter']} & {r['certified']} & {r['bound_value']} & "
            f"{r['target']} & {r['n_calibration']} & {r['passed']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_certificate_table(
    badge_payload: dict[str, Any],
    out_dir: str | Path,
    *,
    name: str = "certificate_table",
) -> dict[str, Path]:
    """Write certificate table in markdown + latex side-by-side."""
    rows = build_certificate_rows(badge_payload)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"{name}.md"
    tex_path = out / f"{name}.tex"
    md_path.write_text(render_markdown(rows))
    tex_path.write_text(render_latex(rows))
    return {"markdown": md_path, "latex": tex_path}


__all__ = [
    "build_certificate_rows",
    "render_latex",
    "render_markdown",
    "write_certificate_table",
]
