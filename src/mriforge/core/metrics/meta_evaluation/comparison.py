"""Side-by-side comparison of meta-evaluation summaries (e.g. real vs betting).

Pure-logic core for ``scripts/sim2rank/compare_meta_eval.py``. Given several
labelled :meth:`MetaEvaluationOutput.to_summary` dicts — typically the two
certification variants of ONE ``eval_mode`` (the powered-only *real* run and the
``--betting`` run) — it reports:

* **rank agreement** (Spearman of ``final_scores``) between the arms. Betting is
  *additive* (it does not feed the aggregation), so real-vs-betting agreement
  must be ``1.0``; a value below ``1`` is a regression signal, not a finding.
* **defensibility-set agreement** (Jaccard of the metrics each arm flags).
* the **betting-vs-powered certification delta** — the directed pairs the L2⁺
  anytime-valid betting order certifies vs. the pooled L2 Hoeffding-powered gate
  (overlap / betting-only / powered-only).

A *failed* arm (e.g. a fail-loud ``eval_mode=3d`` run that wrote no summary) is
passed as ``None`` and reported under ``missing`` rather than crashing the
comparison — so "compare the 3-D arms together" honestly reports that both
variants fail-loud instead of raising. NumPy-free / torch-free: pure ``math``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "compare_summaries",
    "render_comparison_markdown",
    "spearman_rank_correlation",
]


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based average (tie-corrected) ranks of ``values`` (ascending)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of the 1-based positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rank_correlation(
    scores_a: Mapping[str, float], scores_b: Mapping[str, float]
) -> float | None:
    """Spearman ρ between two metric→score maps over their shared metrics.

    Returns ``None`` when fewer than two metrics are shared (ρ undefined). A
    constant score vector on either side yields ``1.0`` when both are constant,
    else ``0.0`` (the degenerate-variance convention).
    """
    # crash→NaN contract: drop any metric whose score is non-finite on either side
    # before ranking. A NaN score corrupts the rank order (its sort position is
    # undefined) and silently yields a spurious rho = 1.0 — a fake "rankings agree"
    # that would mask a genuine divergence in the betting-is-additive check.
    shared = sorted(
        k
        for k in set(scores_a) & set(scores_b)
        if math.isfinite(scores_a[k]) and math.isfinite(scores_b[k])
    )
    n = len(shared)
    if n < 2:
        return None
    ra = _average_ranks([float(scores_a[k]) for k in shared])
    rb = _average_ranks([float(scores_b[k]) for k in shared])
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    var_a = sum((ra[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((rb[i] - mean_b) ** 2 for i in range(n))
    if var_a == 0.0 or var_b == 0.0:
        return 1.0 if (var_a == 0.0 and var_b == 0.0) else 0.0
    return cov / (var_a * var_b) ** 0.5


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _directed_pairs(raw) -> set[tuple[str, str]]:
    """Normalise a JSON list of ``[better, worse]`` pairs to a set of tuples."""
    out: set[tuple[str, str]] = set()
    for pair in raw or []:
        if len(pair) >= 2:
            out.add((str(pair[0]), str(pair[1])))
    return out


def _powered_union(summary: Mapping) -> set[tuple[str, str]]:
    """Pool the per-method Hoeffding-powered certified pairs into one set."""
    powered = summary.get("powered") or {}
    union: set[tuple[str, str]] = set()
    for block in powered.values():
        union |= _directed_pairs(block.get("certified"))
    return union


def _betting_pairs(summary: Mapping) -> set[tuple[str, str]]:
    betting = summary.get("betting") or {}
    pairs: set[tuple[str, str]] = set()
    for block in betting.values():
        pairs |= _directed_pairs(block.get("certified"))
    return pairs


def _arm_view(label: str, summary: Mapping, top_k: int) -> dict:
    final_ranks = list(summary.get("final_ranks") or [])
    defensible = {m for m, flag in (summary.get("defensibility_flags") or {}).items() if flag}
    powered = _powered_union(summary)
    betting = _betting_pairs(summary)
    return {
        "label": label,
        "eval_mode": summary.get("eval_mode"),
        "n_metrics": len(final_ranks),
        "n_samples": summary.get("n_samples"),
        "top_k": final_ranks[:top_k],
        "n_defensible": len(defensible),
        "defensible": sorted(defensible),
        "n_powered_certified": len(powered),
        "n_betting_certified": len(betting),
        "has_betting": bool(betting),
    }


def compare_summaries(
    arms: Sequence[tuple[str, Mapping | None]],
    *,
    top_k: int = 10,
) -> dict:
    """Compare labelled meta-evaluation summaries that share one ``eval_mode``.

    ``arms`` is a list of ``(label, summary_or_None)``; ``None`` marks an arm that
    produced no summary (e.g. the fail-loud 3-D guard). Returns a JSON-able dict
    with per-arm views, cross-arm rank/defensibility agreement, and the
    betting-vs-powered certification delta.
    """
    present = [(lbl, s) for lbl, s in arms if s is not None]
    missing = [lbl for lbl, s in arms if s is None]

    eval_modes = sorted({str(s.get("eval_mode")) for _, s in present})
    consensus_eval_mode = eval_modes[0] if len(eval_modes) == 1 else None

    views = [_arm_view(lbl, s, top_k) for lbl, s in present]

    # Pairwise cross-arm agreement (rank + defensibility-set).
    pairwise: list[dict] = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            (la, sa), (lb, sb) = present[i], present[j]
            rho = spearman_rank_correlation(
                sa.get("final_scores") or {}, sb.get("final_scores") or {}
            )
            def_a = {m for m, f in (sa.get("defensibility_flags") or {}).items() if f}
            def_b = {m for m, f in (sb.get("defensibility_flags") or {}).items() if f}
            pairwise.append(
                {
                    "arms": [la, lb],
                    "spearman_final_scores": rho,
                    # isclose, not ``== 1.0``: identical rankings can compute to
                    # 0.999999… under floating-point averaging of tied ranks, which
                    # would spuriously report "RANKING DIVERGED" (critique L2).
                    # ``rho`` is None when < 2 metrics are shared (undefined).
                    "rank_identical": rho is not None and math.isclose(rho, 1.0, abs_tol=1e-9),
                    "defensible_jaccard": _jaccard(def_a, def_b),
                    "defensible_only": {
                        la: sorted(def_a - def_b),
                        lb: sorted(def_b - def_a),
                    },
                }
            )

    # Betting-vs-powered certification delta (within whichever arm has betting).
    betting_vs_powered: dict | None = None
    for lbl, s in present:
        bpairs = _betting_pairs(s)
        if bpairs:
            ppairs = _powered_union(s)
            betting_vs_powered = {
                "arm": lbl,
                "n_betting": len(bpairs),
                "n_powered": len(ppairs),
                "n_overlap": len(bpairs & ppairs),
                "betting_only": sorted(map(list, bpairs - ppairs)),
                "powered_only": sorted(map(list, ppairs - bpairs)),
            }
            break

    return {
        "consensus_eval_mode": consensus_eval_mode,
        "eval_modes": eval_modes,
        "n_arms": len(arms),
        "n_present": len(present),
        "missing_arms": missing,
        "arms": views,
        "pairwise": pairwise,
        "betting_vs_powered": betting_vs_powered,
    }


def render_comparison_markdown(comparison: Mapping) -> str:
    """Render :func:`compare_summaries` output as a readable markdown report."""
    mode = (
        comparison.get("consensus_eval_mode") or "/".join(comparison.get("eval_modes") or []) or "?"
    )
    lines: list[str] = [f"# meta-evaluate comparison — eval_mode={mode}", ""]

    missing = comparison.get("missing_arms") or []
    if missing:
        lines += [
            f"> **{len(missing)} arm(s) produced no summary:** `{', '.join(missing)}`.",
            "> For `eval_mode=3d` this is the expected fail-loud guard "
            "(NotImplementedError) — there is no leaderboard to compare.",
            "",
        ]

    views = comparison.get("arms") or []
    if views:
        lines += [
            "## Per-arm leaderboard",
            "",
            "| arm | metrics | N | defensible | powered-cert | betting-cert | top-"
            f"{len(views[0].get('top_k') or [])} |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for v in views:
            top = ", ".join(v.get("top_k") or [])
            lines.append(
                f"| {v['label']} | {v['n_metrics']} | {v.get('n_samples')} | "
                f"{v['n_defensible']} | {v['n_powered_certified']} | "
                f"{v['n_betting_certified']} | {top} |"
            )
        lines.append("")

    for pw in comparison.get("pairwise") or []:
        a, b = pw["arms"]
        rho = pw["spearman_final_scores"]
        rho_s = "n/a" if rho is None else f"{rho:.4f}"
        verdict = (
            "identical ranking (betting is additive ✓)"
            if pw["rank_identical"]
            else "RANKING DIVERGED — investigate"
        )
        lines += [
            f"## {a} vs {b}",
            "",
            f"- Spearman(final_scores) = **{rho_s}** — {verdict}",
            f"- defensible-set Jaccard = {pw['defensible_jaccard']:.3f}",
        ]
        for arm_lbl, only in (pw.get("defensible_only") or {}).items():
            if only:
                lines.append(f"  - defensible only in `{arm_lbl}`: {', '.join(only)}")
        lines.append("")

    bvp = comparison.get("betting_vs_powered")
    if bvp:
        lines += [
            "## Betting (L2⁺ anytime-valid) vs powered (L2 Hoeffding)",
            "",
            f"- arm: `{bvp['arm']}` · betting-certified pairs: {bvp['n_betting']} · "
            f"powered-certified (pooled): {bvp['n_powered']} · overlap: "
            f"{bvp['n_overlap']}",
            f"- betting-only pairs: {len(bvp['betting_only'])}",
            f"- powered-only pairs: {len(bvp['powered_only'])}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
