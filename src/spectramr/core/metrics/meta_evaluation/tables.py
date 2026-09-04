"""CSV + LaTeX exporters for ``MetaEvaluationOutput``.

The machine-readable counterpart of the diagnostic figures, plus one
paper-ready LaTeX leaderboard:

* ``summary.csv``        — one row per metric, **rank-ordered**: rank,
                            display name, normalized [0,1] score, raw final
                            z, optimization direction, defensibility, and the
                            per-method z-scores.
* ``per_method_scores.csv`` — one row per metric: raw per-method scores
                              (pre-z-scoring), useful for sanity-checks.
* ``per_family.csv``     — one row per (metric, family): mean / std /
                            min / max / range / spearman with severity.
* ``correlation.csv``    — Pearson correlation matrix between metrics
                            (square table, named index + columns).
* ``raw_long.csv``       — fully unrolled long-format table, one row per
                            (sample, metric): content_id, family, theta,
                            seed, metric_name, value, residual_energy.
* ``leaderboard.tex``    — booktabs LaTeX table of the ranked leaderboard
                            (rank, metric, normalized [0,1], raw z,
                            defensible) ready to ``\\input`` into a paper.

The writers use only the Python ``csv`` module — no pandas — to keep
the dependency footprint identical to the rest of the framework.

Number formatting: every finite numeric cell is rounded to a fixed number
of decimal places (``_fmt``) so a professor opening the CSV in Excel sees
``0.123`` rather than a 15-digit float dump. NaN renders as an empty cell.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from ._report_style import human_label
from .normalization import direction_aware_minmax
from .pipeline import MetaEvaluationOutput

logger = logging.getLogger(__name__)

# Decimal places for rounded numeric cells in the CSV / LaTeX tables.
_PLACES = 4


class TableWriteError(RuntimeError):
    """A table in the suite failed to write.

    Raised by :func:`write_all_tables` in strict mode. A missing results file is
    invisible -- nobody notices the table that *isn't* there -- so a write failure
    must stop the run rather than silently thin the output directory.
    """


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(v: object) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


def _fmt(v: object, places: int = _PLACES) -> str:
    """Render a numeric cell: finite -> rounded string, NaN/None -> ''."""
    f = _safe_float(v)
    if math.isnan(f):
        return ""
    return f"{f:.{places}f}"


def _direction_lookup() -> dict[str, bool]:
    """Best-effort metric -> higher_is_better map.

    Starts from the SSOT :data:`metric_directions.METRIC_HIGHER_IS_BETTER`
    and augments with the registry's per-class direction (set at
    registration, so it covers self-declared metrics too). Never raises —
    the column is informational, so an unknown metric simply has no entry.
    """
    table: dict[str, bool] = {}
    try:
        from ..metric_directions import METRIC_HIGHER_IS_BETTER

        table.update(METRIC_HIGHER_IS_BETTER)
    except Exception:
        pass
    try:
        from ..registry import MetricsRegistry

        for name, cls in MetricsRegistry._metrics.items():
            hib = getattr(cls, "higher_is_better", None)
            if isinstance(hib, bool):
                table.setdefault(name, hib)
    except Exception:
        pass
    return table


def _ranked_metric_order(agg) -> list[str]:
    """Metric names ordered by final rank (1 first); un-ranked metrics last."""
    final = agg.final_scores or {}
    names = (
        list(final.keys()) if final else sorted({m for d in agg.per_method_z.values() for m in d})
    )
    ranking = {n: i + 1 for i, n in enumerate(agg.final_ranks or [])}
    big = len(names) + 1
    return sorted(names, key=lambda n: (ranking.get(n, big), n))


def write_summary_csv(output: MetaEvaluationOutput, path: Path) -> Path:
    """Final consensus table — rank-ordered, rounded, professor-readable.

    Columns: ``rank``, ``metric`` (registry key, for joins), ``display_name``
    (human-readable), ``score_norm`` (direction-aware min-max of the consensus
    z to ``[0, 1]``; 1.0 = best), ``final_z`` (raw), ``higher_is_better``,
    ``defensible``, then one ``z_<method>`` column per ranker.
    """
    path = _ensure_parent(path)
    agg = output.aggregate
    methods = list(agg.per_method_z.keys())
    final = agg.final_scores or {}
    flags = agg.defensibility_flags or {}
    ranking = {n: i + 1 for i, n in enumerate(agg.final_ranks or [])}
    metric_names = _ranked_metric_order(agg)
    # Consensus z is already direction-normalized (higher = better), so no
    # per-metric direction is needed to normalize it to [0, 1].
    score_norm = direction_aware_minmax(final, higher_is_better=None)
    directions = _direction_lookup()

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        header = [
            "rank",
            "metric",
            "display_name",
            "score_norm",
            "final_z",
            "higher_is_better",
            "defensible",
        ] + [f"z_{m}" for m in methods]
        w.writerow(header)
        for name in metric_names:
            hib = directions.get(name)
            row: list[object] = [
                ranking.get(name, ""),
                name,
                human_label(name),
                _fmt(score_norm.get(name)),
                _fmt(final.get(name)),
                "" if hib is None else bool(hib),
                bool(flags.get(name, False)),
            ]
            for m in methods:
                row.append(_fmt(agg.per_method_z.get(m, {}).get(name)))
            w.writerow(row)
    return path


def write_per_method_scores_csv(output: MetaEvaluationOutput, path: Path) -> Path:
    """Raw (pre-z) per-method per-metric scores."""
    path = _ensure_parent(path)
    methods = [r.method for r in output.per_method]
    all_metrics = sorted({m for r in output.per_method for m in (r.scores or {}).keys()})
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "display_name"] + methods)
        for name in all_metrics:
            row: list[object] = [name, human_label(name)]
            for r in output.per_method:
                row.append(_fmt((r.scores or {}).get(name)))
            w.writerow(row)
    return path


def _per_metric_family_stats(
    output: MetaEvaluationOutput, metric_name: str
) -> list[dict[str, object]]:
    """For a single metric, compute per-family summary stats + monotonicity.

    Returns one dict per (metric, family). Spearman of (theta, value) is
    computed across all samples in that family (i.e. across content too)
    so it's the cleanest "does this metric monotonically follow severity
    on this family?" estimator we have.
    """
    dataset = output.dataset
    families = sorted({s.degradation_family for s in dataset.samples})
    values = dataset.metric_values.get(metric_name, [])
    by_family: dict[str, list[tuple[float, float]]] = {f: [] for f in families}
    for s, v in zip(dataset.samples, values, strict=True):
        fv = _safe_float(v)
        if math.isnan(fv):
            continue
        by_family[s.degradation_family].append((float(s.severity.get("theta", 0.0)), fv))

    out: list[dict[str, object]] = []
    for fam in families:
        rows = by_family[fam]
        if not rows:
            out.append(
                {
                    "metric": metric_name,
                    "family": fam,
                    "n": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "range": float("nan"),
                    "spearman_theta": float("nan"),
                }
            )
            continue
        thetas = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n, 1)
        std = math.sqrt(var)
        vmin, vmax = min(vals), max(vals)
        out.append(
            {
                "metric": metric_name,
                "family": fam,
                "n": n,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "range": vmax - vmin,
                "spearman_theta": _spearman_pairs(thetas, vals),
            }
        )
    return out


def _spearman_pairs(x: list[float], y: list[float]) -> float:
    """Plain Spearman; uses average ranks for ties."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    n = len(x)

    def _avg_ranks(seq: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: seq[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and seq[order[j + 1]] == seq[order[i]]:
                j += 1
            avg = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _avg_ranks(x)
    ry = _avg_ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def write_per_family_csv(output: MetaEvaluationOutput, path: Path) -> Path:
    """One row per (metric, family) with mean/std/range/Spearman."""
    path = _ensure_parent(path)
    metric_names = sorted(output.dataset.metric_values.keys())
    rows: list[dict[str, object]] = []
    for name in metric_names:
        rows.extend(_per_metric_family_stats(output, name))
    if not rows:
        path.write_text("metric,family,n,mean,std,min,max,range,spearman_theta\n")
        return path
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r2 = {k: (_fmt(v) if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r2)
    return path


def write_correlation_csv(output: MetaEvaluationOutput, path: Path) -> Path:
    """Pearson correlation matrix between metrics across the full dataset."""
    from .figures import (
        _pearson_corr_matrix,
    )  # local import: keep tables module dep-free

    path = _ensure_parent(path)
    names, corr = _pearson_corr_matrix(output.dataset.metric_values)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        for i, n in enumerate(names):
            row: list[object] = [n]
            for j in range(len(names)):
                row.append(_fmt(corr[i, j]))
            w.writerow(row)
    return path


def write_raw_long_csv(output: MetaEvaluationOutput, path: Path) -> Path:
    """Long-format dump: one row per (sample, metric).

    Useful for pivoting in pandas / R afterwards. Carries the full
    sample identity (content_id, family, theta, seed) so the same CSV
    can drive any plot the framework figures emit.
    """
    path = _ensure_parent(path)
    dataset = output.dataset
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "content_id",
                "degradation_family",
                "theta",
                "seed",
                "metric",
                "value",
                "residual_energy",
            ]
        )
        residual_energies = dataset.residual_energies().tolist()
        for i, s in enumerate(dataset.samples):
            re = float(residual_energies[i]) if i < len(residual_energies) else float("nan")
            theta = float(s.severity.get("theta", 0.0))
            for name, vals in dataset.metric_values.items():
                if i >= len(vals):
                    continue
                w.writerow(
                    [
                        s.content_id,
                        s.degradation_family,
                        theta,
                        s.seed,
                        name,
                        _safe_float(vals[i]),
                        re,
                    ]
                )
    return path


# ---------------------------------------------------------------------------
# LaTeX leaderboard — paper-ready
# ---------------------------------------------------------------------------


def _latex_escape(s: str) -> str:
    """Escape the LaTeX special characters that appear in metric names."""
    out = s
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(a, b)
    return out


def write_leaderboard_latex(
    output: MetaEvaluationOutput, path: Path, *, top_n: int | None = None
) -> Path:
    """Booktabs LaTeX leaderboard, rank-ordered, ready to ``\\input``.

    Columns: rank, metric (display name), normalized score in ``[0, 1]``
    (direction-aware min-max of the consensus z; 1.0 = best in cohort), the
    raw consensus z, and a defensibility check (``\\checkmark`` / ``\\times``).
    ``top_n`` truncates to the leading rows (e.g. a "top-15" slide table);
    ``None`` emits the full cohort.
    """
    path = _ensure_parent(path)
    agg = output.aggregate
    final = agg.final_scores or {}
    flags = agg.defensibility_flags or {}
    ranking = {n: i + 1 for i, n in enumerate(agg.final_ranks or [])}
    score_norm = direction_aware_minmax(final, higher_is_better=None)
    order = _ranked_metric_order(agg)
    if top_n is not None:
        order = order[:top_n]

    lines: list[str] = []
    lines.append("% Auto-generated by spectramr.core.metrics.meta_evaluation.tables")
    lines.append("% Normalized score = direction-aware min-max of the consensus z")
    lines.append("%   (1.0 = best metric in the cohort).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{rlccc}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Metric & Normalized & Consensus $z$ & Defensible \\")
    lines.append(r"\midrule")
    for name in order:
        rank = ranking.get(name, "")
        disp = _latex_escape(human_label(name))
        norm = _fmt(score_norm.get(name), places=3)
        raw = _fmt(final.get(name), places=3)
        mark = r"\checkmark" if flags.get(name, False) else r"$\times$"
        lines.append(f"{rank} & {disp} & {norm} & {raw} & {mark} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Metric reliability leaderboard from the sim2rank "
        r"six-method meta-evaluation. The normalized score is the "
        r"direction-aware min-max of the consensus $z$-score "
        r"($1.0=$ best in cohort); a defensible metric passes the "
        r"six-axis asymmetric rule.}"
    )
    lines.append(r"\label{tab:sim2rank_leaderboard}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def write_all_tables(
    output: MetaEvaluationOutput, out_dir: Path, *, strict: bool = True
) -> list[Path]:
    """Dump the full CSV + LaTeX suite into ``out_dir``. Returns paths written.

    Args:
        strict: raise if any table failed to write (the default). Every table
            failure used to be swallowed with ``except Exception: continue``, so a
            table could vanish from the results directory and the sweep would still
            report success -- and the *absence* of a file is exactly what nobody
            notices. Pass ``strict=False`` only to collect the full failure list in
            one pass; the failures are logged at ERROR either way.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: list[tuple[str, Exception]] = []
    for name, fn in (
        ("summary.csv", write_summary_csv),
        ("per_method_scores.csv", write_per_method_scores_csv),
        ("per_family.csv", write_per_family_csv),
        ("correlation.csv", write_correlation_csv),
        ("raw_long.csv", write_raw_long_csv),
        ("leaderboard.tex", write_leaderboard_latex),
    ):
        try:
            written.append(fn(output, out_dir / name))
        except Exception as exc:  # reported, not swallowed
            logger.error("table %s failed: %s: %s", name, type(exc).__name__, exc)
            failures.append((name, exc))

    if failures and strict:
        detail = "\n".join(f"  {n}: {type(e).__name__}: {e}" for n, e in failures)
        raise TableWriteError(
            f"{len(failures)} of 6 tables failed to write into {out_dir}:\n{detail}"
        ) from failures[0][1]
    return written
