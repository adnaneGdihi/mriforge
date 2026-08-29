"""Gen-2 pairwise sign-agreement d' ranker (registered under the legacy key ``bt``).

.. warning:: **THERE IS NO BRADLEY-TERRY MODEL HERE.** No pairwise comparison
   matrix, no per-item strength vector :math:`\\pi_k`, no MLE over strengths — the
   statistic is a **sign-agreement rate** mapped onto the signal-detection d'
   scale. The 2026-05 audit already established that the α grid the original code
   searched was dead (accuracy at threshold 0.5 is invariant in α > 0). The
   registry key ``bt``, the class name and the ``sub_scores["BT"]`` key survive
   only for backward compatibility with existing configs and reports; read every
   "BT" in this module as *"pairwise sign-agreement rate (d')"*. The genuinely
   Bradley-Terry-shaped quantity in the framework is
   ``meta_eval.compute_bt_holdout_loglik``, which does fit an α by MLE.

Faithful in-core port of ``scripts/sim2rank/meta_eval.py::compute_bt_score``
(the ``BTResult.d_prime`` per metric).

Per metric :math:`k`, for a fixed list of physics-anchored severity pairs
:math:`(a, b)` with :math:`s_a < s_b` (cleaner first), the score is

.. math::
    \\mathrm{accuracy}_k = \\frac{1}{n_\\mathrm{pairs}} \\sum_{(a,b)}
        \\Bigl( \\mathbb{1}\\bigl[\\Delta_k > 0\\bigr]
              + \\tfrac12\\,\\mathbb{1}\\bigl[\\Delta_k = 0\\bigr] \\Bigr),
    \\qquad
    d_k' = \\sqrt{2}\\,\\Phi^{-1}(\\mathrm{accuracy}_k),

with :math:`\\Delta_k = \\widetilde m_k(a) - \\widetilde m_k(b)` and
:math:`\\widetilde m_k` the trajectory oriented so the *better* direction is
positive (sign-flipped when the metric is lower-is-better).

**Ties count ½ — the Mann-Whitney convention.** ``mean(diffs > 0)`` scored an exact
tie as an *error*, punishing a metric for being *indifferent* exactly as hard as
for being *wrong*. A saturating metric that is never wrong but plateaus (13 correct,
15 tied, **0 wrong** of 28 pairs) used to score below chance and lose to a scrambled
metric that is **actively wrong** on 8 of 28 pairs; under the ½ convention the
never-wrong metric wins (d' 0.876 vs 0.842), which is the point of a
discriminability score.

The pair list is drawn **once** with ``np.random.default_rng(seed)`` and shared
across every metric, so the rng stream depends on the **family iteration order**.
This port iterates families in the sorted order returned by
:func:`_build_trajectory_table` and draws pairs over that order — the
faithfulness test feeds the legacy function an ``OrderedDict`` with the matching
axis order so the two rng streams align bit-for-bit.

The per-axis trajectory for a metric is the content-averaged per-family
trajectory (:func:`_per_family_average`) along the sorted severity grid, matching
the ``(M, T)`` ``axis_trajectories`` contract of the legacy function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import _build_trajectory_table, _per_family_average


@dataclass
class BTConfig:
    """Configuration for the pairwise sign-agreement ranker.

    ``n_pairs_per_axis`` and ``seed`` mirror the legacy ``compute_bt_score``
    defaults — the rng stream (and therefore the score) is reproducible only
    when both match the legacy call exactly.
    """

    n_pairs_per_axis: int = 256
    seed: int = 0


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


def _phi_inv_arr(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Φ⁻¹) over an array — legacy parity helper.

    Sole owner of the clip-and-ppf rule in this module: clip to
    ``[1e-6, 1 - 1e-6]`` then ``scipy.stats.norm.ppf``, mirroring
    ``scripts/sim2rank/meta_eval.py::_phi_inv`` exactly. ``norm.ppf`` pays a
    fixed ``argsreduce``/``broadcast_arrays`` cost per *call*, not per element,
    so ranking a metric set issues one call here rather than one per metric.

    Layering (non-negotiable 5) forbids ``src/`` importing the script-side
    twin, so single-ownership of this rule is per-module by construction; the
    two are pinned to each other by the cross-implementation parity test
    ``tests/unit/meta_evaluation/test_bradley_terry.py``.
    """
    from scipy.stats import norm

    return np.asarray(norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6)), dtype=np.float64)


def _phi_inv(p: float) -> float:
    """Scalar Φ⁻¹ — delegates to :func:`_phi_inv_arr` so the rule has one owner."""
    return float(_phi_inv_arr(np.asarray(p, dtype=np.float64)))


@register_ranker(
    "bt",
    generation=2,
    description="Pairwise sign-agreement rate (d-prime); NOT a Bradley-Terry model",
)
class BTRanker(BaseRanker):
    """Pairwise sign-agreement discriminability d' per metric.

    A faithful re-expression of ``compute_bt_score`` against the in-core
    trajectory table. ``scores[name] == BTResult.d_prime[k]`` for the matching
    metric/axis ordering (asserted at ``atol <= 1e-9`` in the paired test).

    Not a Bradley-Terry model — see the module docstring. Ties count ½.
    """

    method_name = "BT"

    def __init__(self, config: BTConfig | None = None) -> None:
        self.config = config or BTConfig()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)

        names = metric_set.names()

        # ── Resolve the shared (M, T) axis trajectories and severity grid ──
        # Build the per-family averaged trajectory for every metric over the
        # SAME family order, so the family list (the rng "axes") is shared
        # across metrics. The severity grid is identical for every metric (the
        # sorted theta grid), so we read it from the first metric.
        family_traj: dict[str, dict[str, np.ndarray]] = {}
        families_order: list[str] = []
        severities: list[float] = []
        for name in names:
            traj, sev, contents, families = _build_trajectory_table(dataset, name)
            if not families_order:
                families_order = families
                severities = sev
            fam_avgs = _per_family_average(traj, families, contents)
            family_traj[name] = {
                fam: t.detach().cpu().numpy().astype(np.float64) for fam, t in fam_avgs.items()
            }

        T = len(severities)
        sev_grid = np.asarray(severities, dtype=np.float64)

        # ── Build the pair list ONCE, shared across metrics (rng parity) ──
        # Iterate families in the resolved order; the rng stream therefore
        # depends on that order — see module docstring.
        rng = np.random.default_rng(self.config.seed)
        pairs: list[tuple[str, int, int]] = []
        for fam in families_order:
            for _ in range(self.config.n_pairs_per_axis):
                a, b = rng.choice(T, size=2, replace=False)
                if sev_grid[a] == sev_grid[b]:
                    continue
                if sev_grid[a] > sev_grid[b]:
                    a, b = b, a
                pairs.append((fam, int(a), int(b)))
        n_pairs = len(pairs)

        if n_pairs == 0:
            scores = dict.fromkeys(names, 0.0)
            accuracy = dict.fromkeys(names, 0.0)
            log_lik = dict.fromkeys(names, -math.log(2))
            return RankingResult(
                method=self.method_name,
                scores=scores,
                ranks=order_from_scores(scores),
                diagnostics={
                    "accuracy": accuracy,
                    "log_likelihood": log_lik,
                    "n_ties": dict.fromkeys(names, 0.0),
                    "n_pairs": 0,
                },
            )

        accuracy: dict[str, float] = {}
        log_lik: dict[str, float] = {}
        n_ties: dict[str, float] = {}

        # ── Build every metric's oriented differences in one pass ─────────
        # The scalar original walked the (metric x pair) grid one element at a
        # time -- at production size 139 metrics x 7,680 pairs = 1.07M
        # interpreted iterations, each one a dict lookup, two NumPy scalar
        # reads and a float() box. Group the pair list by family ONCE, then
        # gather each family's block for all metrics with a single
        # fancy-index. The pair list itself is left untouched: the rng stream
        # depends on family order (see the module docstring), so reordering it
        # would change results.
        fam_pair_idx: dict[str, list[int]] = {}
        for _i, (_fam, _a, _b) in enumerate(pairs):
            fam_pair_idx.setdefault(_fam, []).append(_i)

        diffs_all = np.empty((len(names), n_pairs), dtype=np.float64)
        for _fam, _idxs in fam_pair_idx.items():
            _cols = np.asarray(_idxs, dtype=np.intp)
            _rows_a = np.fromiter((pairs[i][1] for i in _idxs), dtype=np.intp, count=len(_idxs))
            _rows_b = np.fromiter((pairs[i][2] for i in _idxs), dtype=np.intp, count=len(_idxs))
            _mat = np.stack([family_traj[_n][_fam] for _n in names])
            diffs_all[:, _cols] = _mat[:, _rows_a] - _mat[:, _rows_b]

        # ``higher_is_better`` negates the whole row; IEEE negation is exact,
        # so this reproduces the scalar ``d if higher else -d`` bit for bit.
        _higher = np.fromiter(
            (metric_set.is_higher_better(_n) for _n in names), dtype=bool, count=len(names)
        )
        diffs_all *= np.where(_higher, 1.0, -1.0)[:, None]

        # Each ``diffs_all[k]`` is a contiguous 1-D view, so the reductions
        # below see exactly the buffer the scalar loop used to build.
        acc_arr = np.empty(len(names), dtype=np.float64)
        for _k, name in enumerate(names):
            diffs = diffs_all[_k]

            ties = float(np.sum(np.isclose(diffs, 0.0, rtol=0.0, atol=1e-12)))
            n_ties[name] = ties

            std = float(np.std(diffs))
            if std < 1e-12:
                acc = 0.5
                ll = -math.log(2)
            else:
                diffs_n = diffs / (std + 1e-12)
                # Mann-Whitney tie convention: a pair the metric cannot separate is
                # HALF A WIN, not a loss. Scoring a tie as an error made a metric
                # that is never wrong but often indifferent lose to one that is
                # actively wrong (see the module docstring).
                wins = float(np.sum(diffs_n > 0.0))
                acc = (wins + 0.5 * ties) / n_pairs
                ll = float(np.mean(-np.log1p(np.exp(-diffs_n))))
            accuracy[name] = acc
            acc_arr[_k] = acc
            log_lik[name] = ll

        # One batched Φ⁻¹ for the whole metric set instead of one scipy call
        # per metric -- see :func:`_phi_inv_arr` for why that is the expensive
        # axis. Values are unchanged: ppf is elementwise.
        scores: dict[str, float] = {
            name: float(_v)
            for name, _v in zip(names, math.sqrt(2.0) * _phi_inv_arr(acc_arr), strict=True)
        }

        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "accuracy": accuracy,
                "log_likelihood": log_lik,
                "n_ties": n_ties,
                "n_pairs": n_pairs,
            },
        )


__all__ = ["BTConfig", "BTRanker"]
