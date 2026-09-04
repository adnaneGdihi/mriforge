"""Reporting surface for the §8 NR-metric validation harness (spec §8.8).

Turns the :meth:`NRMetricValidationUseCase.run_validation` result dict into
consumable artifacts:

* :func:`write_cards_table` — per-metric x per-tier pass/fail + score CSV.
* :func:`render_cards_markdown` — the same as a markdown table (console/docs).
* :func:`render_nr_validation_figures` — diagnostic PNGs (tier pass/fail
  matrix, per-tier score bars, redundancy correlation heatmap). matplotlib is
  lazy-imported with the headless ``Agg`` backend so the table/markdown path
  needs no plotting dependency.

The harness result is a plain JSON-able dict (not a dataclass) so these
functions are decoupled from the orchestrator and trivially testable.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

__all__ = [
    "render_cards_markdown",
    "render_nr_validation_figures",
    "write_cards_table",
    "write_nr_validation_report",
]


def _tier_order(result: dict[str, Any]) -> list[str]:
    """Tier column order: the run order, falling back to first card's keys."""
    tiers = list(result.get("tiers_run") or [])
    if tiers:
        return tiers
    for card in result.get("cards", {}).values():
        return list(card.get("tiers", {}))
    return []


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return "" if math.isnan(x) else f"{x:.4f}"
    return str(x)


def write_cards_table(result: dict[str, Any], path: Path) -> Path:
    """Write the per-metric validation cards as a CSV.

    Columns: ``metric``, ``overall_pass``, then ``<tier>_pass`` /
    ``<tier>_score`` for every tier that was run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tiers = _tier_order(result)
    header = ["metric", "overall_pass"]
    for t in tiers:
        header += [f"{t}_pass", f"{t}_score"]

    cards = result.get("cards", {})
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for metric in sorted(cards):
            card = cards[metric]
            row: list[str] = [metric, str(bool(card.get("overall_pass", False)))]
            ct = card.get("tiers", {})
            for t in tiers:
                tr = ct.get(t)
                if tr is None:
                    row += ["", ""]
                else:
                    row += [str(bool(tr.get("passed", False))), _fmt(tr.get("score"))]
            writer.writerow(row)
    return path


def render_cards_markdown(result: dict[str, Any]) -> str:
    """Render the validation cards as a GitHub-flavoured markdown table."""
    tiers = _tier_order(result)
    cards = result.get("cards", {})
    head = ["metric", "overall", *tiers]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for metric in sorted(cards):
        card = cards[metric]
        cells = [metric, "PASS" if card.get("overall_pass") else "fail"]
        ct = card.get("tiers", {})
        for t in tiers:
            tr = ct.get(t)
            if tr is None:
                cells.append("-")
            else:
                mark = "PASS" if tr.get("passed") else "fail"
                cells.append(f"{mark} ({_fmt(tr.get('score'))})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_nr_validation_figures(result: dict[str, Any], out_dir: Path) -> list[Path]:
    """Render diagnostic PNGs for a validation run; return written paths.

    matplotlib is imported lazily with the ``Agg`` backend (headless-safe). On a
    missing matplotlib the function returns ``[]`` rather than raising — figures
    are an optional artifact, the cards CSV is the source of truth.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:  # pragma: no cover - matplotlib is an optional extra
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tiers = _tier_order(result)
    cards = result.get("cards", {})
    metrics = sorted(cards)
    if not metrics or not tiers:
        return written

    # (1) tier pass/fail matrix (metrics x tiers): 1 = pass, 0 = fail, NaN = absent
    mat = np.full((len(metrics), len(tiers)), np.nan)
    for i, m in enumerate(metrics):
        ct = cards[m].get("tiers", {})
        for j, t in enumerate(tiers):
            tr = ct.get(t)
            if tr is not None:
                mat[i, j] = 1.0 if tr.get("passed") else 0.0
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(tiers), 0.5 + 0.32 * len(metrics)))
    ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_title("NR metric validation — tier pass/fail")
    fig.tight_layout()
    p = out_dir / "nr_cards_matrix.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # (2) per-tier headline score bars
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(tiers), 0.5 + 0.32 * len(metrics)))
    width = 0.8 / max(1, len(tiers))
    y = np.arange(len(metrics))
    for j, t in enumerate(tiers):
        vals = []
        for m in metrics:
            tr = cards[m].get("tiers", {}).get(t)
            s = tr.get("score") if tr else None
            vals.append(
                s
                if isinstance(s, (int, float)) and not (isinstance(s, float) and math.isnan(s))
                else 0.0
            )
        ax.barh(y + j * width, vals, height=width, label=t)
    ax.set_yticks(y + 0.4 - width / 2)
    ax.set_yticklabels(metrics, fontsize=8)
    ax.set_xlabel("tier headline score")
    ax.legend(fontsize=7)
    ax.set_title("NR metric validation — per-tier scores")
    fig.tight_layout()
    p = out_dir / "nr_tier_scores.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # (3) redundancy correlation heatmap (if present)
    redundancy = result.get("redundancy") or {}
    corr = redundancy.get("corr") or {}
    if corr:
        names = sorted({n for pair in corr for n in pair.split("|")})
        idx = {n: i for i, n in enumerate(names)}
        cm = np.full((len(names), len(names)), np.nan)
        for i in range(len(names)):
            cm[i, i] = 1.0
        for pair, v in corr.items():
            a, b = pair.split("|")
            if a in idx and b in idx and isinstance(v, (int, float)):
                cm[idx[a], idx[b]] = v
                cm[idx[b], idx[a]] = v
        fig, ax = plt.subplots(figsize=(0.6 + 0.5 * len(names), 0.6 + 0.5 * len(names)))
        im = ax.imshow(cm, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Redundancy |rho| (§8.4 gate)")
        fig.tight_layout()
        p = out_dir / "nr_redundancy_corr.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)

    return written


def write_nr_validation_report(
    result: dict[str, Any],
    out_dir: Path,
    *,
    figures: bool = False,
) -> list[Path]:
    """Write the cards CSV + markdown (and optionally figures); return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [write_cards_table(result, out_dir / "nr_cards.csv")]
    md = out_dir / "nr_cards.md"
    md.write_text(render_cards_markdown(result), encoding="utf-8")
    written.append(md)
    if figures:
        written += render_nr_validation_figures(result, out_dir)
    return written
