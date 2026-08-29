"""Aggregator + permutation calibration of the defensibility flag (critique §2)."""

from __future__ import annotations

from mriforge.core.metrics.meta_evaluation.aggregation import (
    AggregatorConfig,
    aggregate,
    calibrate_defensibility,
)
from mriforge.core.metrics.meta_evaluation.types import RankingResult


def _six_axis_result(method: str, scores: dict[str, float]) -> RankingResult:
    return RankingResult(
        method=method, scores=scores, ranks=sorted(scores, key=lambda k: -scores[k])
    )


def test_calibration_off_by_default() -> None:
    r = _six_axis_result("CDSCR", {"a": 1.0, "b": 0.0})
    out = aggregate([r], AggregatorConfig(weights={"CDSCR": 1.0}, min_positive_axes=1))
    assert out.calibration is None


def test_strong_unanimous_metric_is_calibrated_significant() -> None:
    # 'winner' is top on all 6 axes; 'a'..'d' are a noisy cloud. With 6 axes the
    # null prob of being non-negative on all 6 by chance is small, so the winner's
    # breadth p-value should clear alpha while the flag stays True.
    methods = ["CDSCR", "LGDR", "SFA", "ESD", "FSPD", "Sim2Rank"]
    pool = ["winner", "a", "b", "c", "d"]
    results = []
    for i, meth in enumerate(methods):
        scores = {"winner": 5.0}
        # deterministic but varied middling scores for the others
        for j, m in enumerate(pool[1:]):
            scores[m] = float(((i + 1) * (j + 1)) % 3) - 1.0
        results.append(_six_axis_result(meth, scores))
    cfg = AggregatorConfig(
        weights=dict.fromkeys(methods, 1.0),
        min_positive_axes=5,
        floor_z=-1.5,
        calibrate=True,
        n_perm=400,
        calibration_seed=0,
    )
    out = aggregate(results, cfg)
    cal = out.calibration
    assert cal is not None
    # The flag fired for the winner...
    assert out.defensibility_flags["winner"] is True
    # ...and it survives calibration (beyond-chance breadth of agreement).
    assert cal.calibrated_flags["winner"] is True
    assert cal.pvalues["winner"] <= 0.05
    # null_fpr is a probability.
    assert 0.0 <= cal.null_fpr <= 1.0


def test_calibration_is_deterministic_under_seed() -> None:
    methods = ["CDSCR", "LGDR", "SFA", "ESD", "FSPD", "Sim2Rank"]
    results = [
        _six_axis_result(m, {"x": float(i) - 2.5, "y": 2.5 - float(i), "z": 0.0})
        for i, m in enumerate(methods)
    ]
    z = {r.method: r.standardize() for r in results}
    flags = dict.fromkeys(["x", "y", "z"], True)
    a = calibrate_defensibility(
        z, flags, min_positive_axes=5, floor_z=-1.0, n_perm=200, seed=7
    )
    b = calibrate_defensibility(
        z, flags, min_positive_axes=5, floor_z=-1.0, n_perm=200, seed=7
    )
    assert a.pvalues == b.pvalues
    assert a.null_fpr == b.null_fpr


def test_empty_pool_calibration_safe() -> None:
    cal = calibrate_defensibility(
        {}, {}, min_positive_axes=5, floor_z=-1.0, n_perm=10, seed=0
    )
    assert cal.null_fpr == 0.0
    assert cal.pvalues == {}


def test_unweighted_present_method_gets_uniform_fallback_not_zero() -> None:
    """A present method with no declared weight must still affect the aggregate.

    Pre-fix ``weights.get(method, 0.0)`` zeroed any ranker (e.g. resolved via
    ``from_registry``) whose name was not in the curated 6-key weight dict — it ran
    at full cost but contributed nothing (a dead axis; critique B1). The uniform
    fallback gives it the mean of the declared weights so it genuinely counts.
    """
    # Two axes: one weighted ("CDSCR"), one UNweighted ("ADR"). They disagree on
    # which metric is best. If ADR were weight-0, "a" would win outright; with the
    # uniform fallback ADR pulls "b" up so the gap closes / flips.
    weighted = _six_axis_result("CDSCR", {"a": 2.0, "b": -2.0})
    unweighted = _six_axis_result("ADR", {"a": -2.0, "b": 2.0})
    cfg = AggregatorConfig(weights={"CDSCR": 0.5})  # ADR absent from the dict
    out = aggregate([weighted, unweighted], cfg)
    # ADR contributed: the two opposite axes cancel, so the scores are ~equal,
    # NOT dominated by CDSCR alone (which would give final["a"] > 0 >> final["b"]).
    assert abs(out.final_scores["a"] - out.final_scores["b"]) < 1e-9


def test_aggregate_stamps_breadth_semantics() -> None:
    """The z-aggregator labels its defensibility flag as 'breadth' (critique A2)."""
    out = aggregate(
        [_six_axis_result("CDSCR", {"a": 1.0, "b": 0.0})],
        AggregatorConfig(weights={"CDSCR": 1.0}, min_positive_axes=1),
    )
    assert out.defensibility_semantics == "breadth"


def _pool_results(n_methods: int, positive_methods: int) -> list[RankingResult]:
    """``n_methods`` rankers; ``target`` is above-mean (z>=0) on the first
    ``positive_methods`` of them and below-mean on the rest."""
    results = []
    for i in range(n_methods):
        hi = i < positive_methods
        scores = {"target": 5.0 if hi else -5.0, "f1": 0.0, "f2": 0.0}
        results.append(
            RankingResult(
                method=f"M{i}",
                scores=scores,
                ranks=sorted(scores, key=lambda k: -scores[k]),
            )
        )
    return results


def test_min_positive_scales_with_ranker_pool_size() -> None:
    # #296: "5 of 6 axes" was an ABSOLUTE 5. from_registry wires ~12 rankers, so
    # absolute-5 silently became "5 of 12" (~42%), not the intended ~83%. The
    # DEFAULT (no min_positive_axes) now scales as ceil(5/6 * n) = 10 at n=12.
    n = 12
    weights = {f"M{i}": 1.0 for i in range(n)}
    cfg = AggregatorConfig(weights=weights, floor_z=-100.0)  # default fraction path

    out5 = aggregate(_pool_results(n, positive_methods=5), cfg)
    assert (
        out5.defensibility_flags["target"] is False
    ), "5 of 12 (~42%) must NOT be defensible under the fraction rule"
    out10 = aggregate(_pool_results(n, positive_methods=10), cfg)
    assert out10.defensibility_flags["target"] is True, "10 of 12 clears ceil(5/6*12)"


def test_explicit_min_positive_axes_stays_absolute() -> None:
    # No silent override (#15): an explicit min_positive_axes is honored as an
    # absolute count regardless of pool size, so pre-existing callers are unchanged.
    n = 12
    weights = {f"M{i}": 1.0 for i in range(n)}
    cfg = AggregatorConfig(weights=weights, min_positive_axes=5, floor_z=-100.0)
    out5 = aggregate(_pool_results(n, positive_methods=5), cfg)
    assert out5.defensibility_flags["target"] is True, "absolute 5 met at 5 of 12"
