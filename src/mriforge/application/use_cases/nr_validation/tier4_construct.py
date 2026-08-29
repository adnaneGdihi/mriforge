"""§8.4 Tier 4 — construct validity + redundancy gate for the NR metric battery.

Three offline sub-checks, all CPU-canonical, deterministic, and crash-safe
(every metric / statistical computation is wrapped; a failure is recorded as
``float('nan')`` rather than raised — mirroring the precompute crash→NaN
contract):

(A) **Test-retest reliability** — ICC(2,1) over two replicate sweeps that share
    the ``(content, family, theta)`` grid but use a *different* seed. A reliable
    metric assigns near-identical scores across replicates; pass at
    ``ICC ≥ 0.75`` (spec §8.4). Implemented as a pure-torch two-way
    random-effects ANOVA over the ``[n, 2]`` finite value matrix — we do *not*
    abuse the registered ``icc_3_1`` metric (which takes ``(pred, target)``
    image tensors, not a value matrix).

(B) **Invariance / null falsification** — a quality metric must be (approximately)
    invariant to a transform that does not change perceived quality. We apply a
    horizontal flip (``torch.flip``) to each clean reference and count a
    *violation* when the relative change of the NR score exceeds ``rel_tol``.
    Pass at zero violations.

(C) **Redundancy gate** — a greedy ``|spearman|`` prune that finalises the
    texture-group membership (spec §11: "the redundancy gate has pruned the
    texture group"). Reported, **not** pass-gated.

The :func:`score_construct_validity` combiner folds (A) and (B) into one
:class:`TierResult` per metric (``tier="construct_validity"``); the redundancy
gate is returned separately by :func:`redundancy_gate`.
"""

from __future__ import annotations

from typing import Any

import torch

from mriforge.application.use_cases.nr_validation.cards import (
    TierResult,
    eval_metric,
    spearman,
)
from mriforge.core.metrics.meta_evaluation.types import MetricEvaluationDataset

__all__ = [
    "icc_2_1",
    "redundancy_gate",
    "score_construct_validity",
    "score_invariance",
    "score_test_retest",
]

_EPS = 1e-12


def icc_2_1(matrix: torch.Tensor) -> float:
    """Two-way random-effects single-measurement ICC(2,1) over a value matrix.

    Computes the intraclass correlation for absolute agreement among ``k``
    replicate measurements of ``n`` subjects, using the spec §8.4 formula::

        ICC(2,1) = (MS_R - MS_E)
                   / (MS_R + (k-1) MS_E + (k / n)(MS_C - MS_E))

    where ``MS_R`` is the between-subjects (rows) mean square, ``MS_C`` the
    between-replicates (columns) mean square, and ``MS_E`` the residual mean
    square of a two-way ANOVA without interaction.

    Args:
        matrix: ``[n, k]`` tensor of measurements (subjects x replicates). Rows
            containing any non-finite entry are dropped before the ANOVA.

    Returns:
        The ICC(2,1) coefficient as a float, or ``float('nan')`` when fewer than
        three complete subjects remain, ``k < 2``, or the denominator collapses.
        Never raises.
    """
    try:
        mat = torch.as_tensor(matrix, dtype=torch.float64)
        if mat.ndim != 2:
            return float("nan")
        finite_rows = torch.isfinite(mat).all(dim=1)
        mat = mat[finite_rows]
        n, k = mat.shape
        if n < 3 or k < 2:
            return float("nan")
        n_f = float(n)
        k_f = float(k)

        grand = mat.mean()
        row_means = mat.mean(dim=1)  # [n]
        col_means = mat.mean(dim=0)  # [k]

        # Sum of squares (two-way, no interaction).
        ss_total = float(((mat - grand) ** 2).sum().item())
        ss_rows = float((k_f * (row_means - grand) ** 2).sum().item())
        ss_cols = float((n_f * (col_means - grand) ** 2).sum().item())
        ss_error = ss_total - ss_rows - ss_cols
        # Guard tiny negative residuals from floating-point cancellation.
        if ss_error < 0.0:
            ss_error = 0.0

        df_rows = n_f - 1.0
        df_cols = k_f - 1.0
        df_error = (n_f - 1.0) * (k_f - 1.0)
        if df_rows <= 0.0 or df_cols <= 0.0 or df_error <= 0.0:
            return float("nan")

        ms_rows = ss_rows / df_rows
        ms_cols = ss_cols / df_cols
        ms_error = ss_error / df_error

        denom = ms_rows + (k_f - 1.0) * ms_error + (k_f / n_f) * (ms_cols - ms_error)
        if abs(denom) < _EPS:
            return float("nan")
        icc = (ms_rows - ms_error) / denom
        if icc != icc:  # NaN guard
            return float("nan")
        return float(icc)
    except Exception:
        return float("nan")


def score_test_retest(
    dataset_a: MetricEvaluationDataset,
    dataset_b: MetricEvaluationDataset,
    nr_names: list[str],
    *,
    icc_threshold: float = 0.75,
) -> dict[str, TierResult]:
    """Sub-check (A) — test-retest reliability via ICC(2,1) per NR metric.

    ``dataset_a`` and ``dataset_b`` are two sweeps over the *same*
    ``(content, family, theta)`` grid with a *different* seed, so their metric
    columns are aligned sample-for-sample. The two aligned columns are the two
    "raters" (replicates) feeding the ICC.

    Args:
        dataset_a: First replicate sweep.
        dataset_b: Second replicate sweep (different seed, same grid → aligned).
        nr_names: NR metric names to score.
        icc_threshold: Pass threshold for ICC(2,1) (spec §8.4 default ``0.75``).

    Returns:
        ``{metric: TierResult}`` with ``tier="test_retest"``,
        ``score = ICC(2,1)``, ``passed = ICC ≥ icc_threshold`` and
        ``detail = {"icc_2_1", "n"}``. A metric missing from either dataset (or
        with too few aligned finite pairs) records NaN and ``passed=False``.
    """
    out: dict[str, TierResult] = {}
    for name in nr_names:
        col_a = dataset_a.metric_values.get(name)
        col_b = dataset_b.metric_values.get(name)
        if col_a is None or col_b is None:
            out[name] = TierResult(
                "test_retest",
                False,
                float("nan"),
                {"reason": "metric not in one of the replicate datasets", "n": 0},
            )
            continue
        try:
            a = torch.as_tensor(col_a, dtype=torch.float64)
            b = torch.as_tensor(col_b, dtype=torch.float64)
            m = min(a.shape[0], b.shape[0])
            a, b = a[:m], b[:m]
            matrix = torch.stack([a, b], dim=1)  # [n, 2]
            finite_rows = torch.isfinite(matrix).all(dim=1)
            n = int(finite_rows.sum().item())
            icc = icc_2_1(matrix)
        except Exception:
            n = 0
            icc = float("nan")
        passed = bool(icc == icc and icc >= icc_threshold)
        out[name] = TierResult(
            "test_retest",
            passed,
            icc,
            {"icc_2_1": icc, "n": n},
        )
    return out


def score_invariance(
    clean_volumes: list[tuple[str, torch.Tensor]],
    nr_names: list[str],
    *,
    rel_tol: float = 0.05,
) -> dict[str, TierResult]:
    """Sub-check (B) — invariance / null falsification under a quality-preserving flip.

    For each clean reference we evaluate the NR metric on the *original* image
    (as both ``pred`` and ``ref``) and on its horizontal flip (likewise as both
    ``pred`` and ``ref``). A horizontal flip does not change perceived image
    quality, so a valid quality metric should be approximately invariant. A
    *violation* is counted when the relative change exceeds ``rel_tol``::

        |m_flip - m_0| / (|m_0| + eps) > rel_tol

    A metric that returns NaN on either evaluation is treated as "no opinion"
    for that reference (not a violation).

    Args:
        clean_volumes: ``[(content_id, tensor)]`` clean references on CPU,
            shaped ``[C, H, W]`` or ``[C, D, H, W]``.
        nr_names: NR metric names to score.
        rel_tol: Relative-change tolerance (spec §8.4 default ``0.05``).

    Returns:
        ``{metric: TierResult}`` with ``tier="invariance"``,
        ``score = violation_fraction``, ``passed = (violations == 0)`` and
        ``detail = {"violations", "n", "max_rel_change"}``. ``n`` counts the
        references on which the metric had an opinion (both evals finite).
        Never raises.
    """
    out: dict[str, TierResult] = {}
    for name in nr_names:
        violations = 0
        n_opinion = 0
        max_rel = 0.0
        for _content_id, vol in clean_volumes:
            try:
                clean = torch.as_tensor(vol)
                flipped = torch.flip(clean, dims=[-1])
                m0 = eval_metric(name, clean, clean)
                mt = eval_metric(name, flipped, flipped)
            except Exception:
                continue
            if not (m0 == m0 and mt == mt):  # NaN on either → no opinion
                continue
            n_opinion += 1
            rel = abs(mt - m0) / (abs(m0) + _EPS)
            if rel > max_rel:
                max_rel = float(rel)
            if rel > rel_tol:
                violations += 1
        score = float(violations) / n_opinion if n_opinion > 0 else float("nan")
        passed = bool(violations == 0 and n_opinion > 0)
        out[name] = TierResult(
            "invariance",
            passed,
            score,
            {
                "violations": int(violations),
                "n": int(n_opinion),
                "max_rel_change": float(max_rel),
            },
        )
    return out


def redundancy_gate(
    dataset: MetricEvaluationDataset,
    nr_names: list[str],
    *,
    corr_threshold: float = 0.95,
) -> dict[str, Any]:
    """Sub-check (C) — greedy ``|spearman|`` redundancy prune (reported, not gated).

    Walks ``nr_names`` in input order and KEEPS a metric unless it is
    ``|rho| > corr_threshold``-correlated with an already-kept metric, in which
    case it is marked PRUNED and recorded as a duplicate of the *first* kept
    metric it duplicates. This finalises the texture-group membership per spec
    §8.4 / §11.

    Args:
        dataset: The sweep whose NR columns are correlated.
        nr_names: NR metric names, evaluated in this order (input order is the
            keep-priority order).
        corr_threshold: ``|rho|`` above which two metrics are deemed redundant
            (spec §8.4 default ``0.95``).

    Returns:
        ``{"kept": [...], "pruned": {metric: duplicates_of}, "corr": {...}}``
        where ``corr`` maps ``"a|b"`` → ``|rho(a, b)|`` for every evaluated pair.
        Never raises; a metric missing from the dataset is skipped (and recorded
        in ``pruned`` as duplicating ``None`` → ``"__absent__"``).
    """
    kept: list[str] = []
    pruned: dict[str, str] = {}
    corr: dict[str, float] = {}

    # Cache per-metric columns as float64 tensors.
    columns: dict[str, torch.Tensor | None] = {}
    for name in nr_names:
        vals = dataset.metric_values.get(name)
        if vals is None:
            columns[name] = None
        else:
            try:
                columns[name] = torch.as_tensor(vals, dtype=torch.float64)
            except Exception:
                columns[name] = None

    for name in nr_names:
        col = columns[name]
        if col is None:
            pruned[name] = "__absent__"
            continue
        duplicate_of: str | None = None
        for kept_name in kept:
            kcol = columns[kept_name]
            if kcol is None:
                continue
            try:
                rho = abs(spearman(col, kcol))
            except Exception:
                rho = float("nan")
            corr[f"{name}|{kept_name}"] = float(rho)
            if rho == rho and rho > corr_threshold:
                duplicate_of = kept_name
                break
        if duplicate_of is None:
            kept.append(name)
        else:
            pruned[name] = duplicate_of

    return {"kept": kept, "pruned": pruned, "corr": corr}


def score_construct_validity(
    dataset_a: MetricEvaluationDataset,
    dataset_b: MetricEvaluationDataset,
    clean_volumes: list[tuple[str, torch.Tensor]],
    nr_names: list[str],
    *,
    icc_threshold: float = 0.75,
    rel_tol: float = 0.05,
) -> dict[str, TierResult]:
    """Combiner — fold (A) test-retest and (B) invariance into one TierResult.

    The redundancy gate (C) is intentionally *not* folded in; it is reported via
    :func:`redundancy_gate` and is not part of the pass/fail (spec §11:
    redundancy is pruned/reported, not pass-gated).

    Args:
        dataset_a: First replicate sweep (for ICC).
        dataset_b: Second replicate sweep, different seed / same grid (for ICC).
        clean_volumes: Clean references for the invariance check.
        nr_names: NR metric names to score.
        icc_threshold: ICC(2,1) pass threshold.
        rel_tol: Invariance relative-change tolerance.

    Returns:
        ``{metric: TierResult}`` with ``tier="construct_validity"``,
        ``score = ICC(2,1)``, ``passed = (ICC passed) AND (invariance passed)``,
        and ``detail`` merging
        ``{"icc_2_1", "n", "invariance_violations", "invariance_passed"}``.
    """
    retest = score_test_retest(dataset_a, dataset_b, nr_names, icc_threshold=icc_threshold)
    invariance = score_invariance(clean_volumes, nr_names, rel_tol=rel_tol)

    out: dict[str, TierResult] = {}
    for name in nr_names:
        r = retest.get(name)
        inv = invariance.get(name)
        icc = r.score if r is not None else float("nan")
        icc_passed = bool(r.passed) if r is not None else False
        n = int(r.detail.get("n", 0)) if r is not None else 0
        inv_passed = bool(inv.passed) if inv is not None else False
        inv_violations = int(inv.detail.get("violations", 0)) if inv is not None else 0
        out[name] = TierResult(
            "construct_validity",
            bool(icc_passed and inv_passed),
            icc,
            {
                "icc_2_1": icc,
                "n": n,
                "invariance_violations": inv_violations,
                "invariance_passed": inv_passed,
            },
        )
    return out
