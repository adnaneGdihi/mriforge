"""Gen-1 statistical rankers — ADR, SCVR, CDRS.

These expose, as standalone registered :class:`BaseRanker` rankers, the three
per-metric statistics the Gen-3 :class:`Sim2RankRanker` already computes and
fuses internally (``rankers/sim2rank.py``):

* **ADR** — Area of Discriminability (per-family monotone discriminability).
* **SCVR** — Signal-to-Content Variance Ratio (two-way ANOVA F-statistic).
* **CDRS** — Clean-to-Degraded Robustness Score.

Reusing the validated in-core helpers (``_build_trajectory_table`` /
``compute_adr`` / ``compute_scvr`` / ``compute_cdrs``) makes these *faithful by
construction* — identical math to the fused Sim2Rank ranker — rather than a
re-port of the ``scripts/sim2rank/scoring.py`` bespoke-input functions. Sim2Rank
fuses the three; the agreement matrix also wants each on its own (multi-
granularity), so they register as ``generation=1``.
"""

from __future__ import annotations

import math

from ..ranking_guards import assert_not_degenerate
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import (
    Sim2RankConfig,
    _build_trajectory_table,
    _content_family_tensor,
    _per_family_average,
    compute_adr,
    compute_cdrs,
    compute_scvr,
)


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


class _StatisticalRanker(BaseRanker):
    """Shared config plumbing for the Gen-1 rankers."""

    def __init__(self, config: Sim2RankConfig | None = None) -> None:
        self.config = config or Sim2RankConfig()


@register_ranker("adr", generation=1, description="Area of Discriminability (per-family monotone)")
class ADRRanker(_StatisticalRanker):
    method_name = "ADR"

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        scores: dict[str, float] = {}
        breakdowns: dict[str, dict[str, float]] = {}
        for name in metric_set.names():
            higher = metric_set.is_higher_better(name)
            traj, _sev, contents, families = _build_trajectory_table(dataset, name)
            family_avgs = _per_family_average(traj, families, contents)
            breakdown = {
                fam: compute_adr(
                    t,
                    epsilon=self.config.epsilon,
                    higher_is_better=higher,
                    nonsat=self.config.nonsat,
                )
                for fam, t in family_avgs.items()
            }
            breakdowns[name] = breakdown
            scores[name] = (
                float(sum(breakdown.values()) / max(len(breakdown), 1)) if breakdown else 0.0
            )
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=self._ranks_from_scores(scores),
            diagnostics={"per_family_adr": breakdowns},
        )


@register_ranker(
    "scvr", generation=1, description="Signal-to-Content Variance Ratio (two-way ANOVA)"
)
class SCVRRanker(_StatisticalRanker):
    method_name = "SCVR"

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        scores: dict[str, float] = {}
        for name in metric_set.names():
            traj, _sev, contents, families = _build_trajectory_table(dataset, name)
            cf_tensor = _content_family_tensor(traj, families, contents)
            scores[name] = compute_scvr(cf_tensor) if cf_tensor.numel() else 0.0
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=self._ranks_from_scores(scores),
            diagnostics={},
        )


@register_ranker("cdrs", generation=1, description="Clean-to-Degraded Robustness Score")
class CDRSRanker(_StatisticalRanker):
    method_name = "CDRS"

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        scores: dict[str, float] = {}
        components: dict[str, tuple[float, float]] = {}
        for name in metric_set.names():
            higher = metric_set.is_higher_better(name)
            traj, severities, contents, families = _build_trajectory_table(dataset, name)
            family_avgs = _per_family_average(traj, families, contents)
            clean_scores: list[float] = []
            if severities:
                for c in contents:
                    for fam in families:
                        arr = traj.get((c, fam))
                        if arr is None:
                            continue
                        v = arr[0]
                        if math.isfinite(v):
                            clean_scores.append(v)
            cdrs, c_m, m_m = compute_cdrs(
                clean_scores, list(family_avgs.values()), higher_is_better=higher
            )
            scores[name] = cdrs
            components[name] = (c_m, m_m)
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=self._ranks_from_scores(scores),
            diagnostics={"cdrs_components": components},
        )


__all__ = ["ADRRanker", "CDRSRanker", "SCVRRanker"]
