"""Base class for meta-evaluation rankers.

A ranker takes a ``MetricSet`` plus ``MetricEvaluationDataset`` and emits a
``RankingResult``. All concrete rankers are expected to:

* compute one scalar score per metric in ``metric_set.metrics``,
* set the score's sign so that higher = better metric,
* emit ``ranks`` sorted descending by score,
* populate ``diagnostics`` with whatever the figure suite needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ranking_guards import order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult


class BaseRanker(ABC):
    method_name: str = "abstract"

    @abstractmethod
    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult: ...

    @staticmethod
    def _ranks_from_scores(scores: dict[str, float]) -> list[str]:
        """Metric names sorted best→worst — see :func:`ranking_guards.order_from_scores`.

        Ties break by **name**, never by dict insertion order, so two rankers that
        both saturate produce orderings that are honestly comparable rather than
        coincidentally identical (#243).

        Non-finite scores are not interchangeable: ``+inf`` is a legitimate best case
        (SCVR with zero content nuisance) and ranks FIRST; ``NaN``/``-inf`` mean the
        metric crashed and rank LAST. The previous implementation sent every
        non-finite score to the bottom, which inverted the ideal-SCVR case (#240).
        """
        return order_from_scores(scores)
