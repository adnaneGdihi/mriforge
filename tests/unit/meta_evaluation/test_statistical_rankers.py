"""Faithfulness tests for the Gen-1 statistical rankers (ADR / SCVR / CDRS).

The Gen-1 rankers expose the per-metric statistics the Gen-3 Sim2Rank fusion
ranker computes internally. They must produce *exactly* those values — that is
the contract that makes the reuse faithful (no re-implementation drift).
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation.rankers import (
    ADRRanker,
    CDRSRanker,
    SCVRRanker,
    Sim2RankRanker,
)
from spectramr.core.metrics.meta_evaluation.simulator import (
    SimulatorConfig,
    precompute_metric_values,
    run_simulator,
)
from spectramr.core.metrics.meta_evaluation.types import MetricEvaluationDataset


def _dataset(metric_set, clean_volumes) -> MetricEvaluationDataset:
    cfg = SimulatorConfig(
        families=["gaussian_noise", "gaussian_blur"], n_severities_per_family=4, seed=0
    )
    samples = run_simulator(clean_volumes, cfg)
    values = precompute_metric_values(samples, metric_set.metrics)
    return MetricEvaluationDataset(samples=samples, metric_values=values)


def test_adr_ranker_matches_sim2rank_internal(metric_set, clean_volumes) -> None:
    ds = _dataset(metric_set, clean_volumes)
    s2r = Sim2RankRanker().rank(metric_set, ds)
    adr = ADRRanker().rank(metric_set, ds)
    assert set(adr.scores) == set(metric_set.names())
    for m in metric_set.names():
        assert adr.scores[m] == pytest.approx(s2r.diagnostics["adr"][m])


def test_scvr_ranker_matches_sim2rank_internal(metric_set, clean_volumes) -> None:
    ds = _dataset(metric_set, clean_volumes)
    s2r = Sim2RankRanker().rank(metric_set, ds)
    scvr = SCVRRanker().rank(metric_set, ds)
    for m in metric_set.names():
        assert scvr.scores[m] == pytest.approx(s2r.diagnostics["scvr"][m])


def test_cdrs_ranker_matches_sim2rank_internal(metric_set, clean_volumes) -> None:
    ds = _dataset(metric_set, clean_volumes)
    s2r = Sim2RankRanker().rank(metric_set, ds)
    cdrs = CDRSRanker().rank(metric_set, ds)
    for m in metric_set.names():
        assert cdrs.scores[m] == pytest.approx(s2r.diagnostics["cdrs"][m])


def test_statistical_rankers_emit_valid_ranking(metric_set, clean_volumes) -> None:
    ds = _dataset(metric_set, clean_volumes)
    for ranker in (ADRRanker(), SCVRRanker(), CDRSRanker()):
        res = ranker.rank(metric_set, ds)
        assert sorted(res.ranks) == sorted(metric_set.names())
        assert res.method in {"ADR", "SCVR", "CDRS"}
