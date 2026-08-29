"""Behavioral tests for each ranker.

The hypothesis is that ``mse`` (a real metric) ranks above ``noise`` and
``constant`` (the null metrics). We do not test exact scores — finite-
sample variance dominates with the small phantom set used in conftest.
"""

from __future__ import annotations

import math

import torch

from mriforge.core.metrics.meta_evaluation.rankers import (
    CDSCRRanker,
    ESDRanker,
    FSPDRanker,
    LGDRRanker,
    SFARanker,
)
from mriforge.core.metrics.meta_evaluation.rankers.base import BaseRanker
from mriforge.core.metrics.meta_evaluation.rankers.sfa import _cosine


def test_ranks_from_scores_orders_finite_descending() -> None:
    assert BaseRanker._ranks_from_scores({"a": 0.1, "b": 0.9, "c": 0.5}) == [
        "b",
        "c",
        "a",
    ]


def test_ranks_from_scores_nan_sorts_last_not_first() -> None:
    # crash→NaN contract: a metric whose score is NaN must NOT be ranked best —
    # it carries no signal and belongs last. (Regression: ``sorted(key=-score)``
    # placed NaN wherever it sat in dict order, so a crashed metric could rank #1.)
    ranks = BaseRanker._ranks_from_scores({"a": float("nan"), "b": 1.0, "c": 0.5})
    assert ranks == ["b", "c", "a"]


def test_ranks_from_scores_nonfinite_tail_is_deterministic() -> None:
    # +inf and NaN are NOT interchangeable (#251). +inf is a legitimate *best*
    # case — SCVR with zero nuisance variance is the ideal metric, not a crash —
    # so it ranks FIRST. Only NaN/-inf mean "the metric crashed" and rank last.
    # Either way the order is insertion-order independent, so it reproduces.
    r1 = BaseRanker._ranks_from_scores({"z": float("nan"), "a": float("inf"), "m": 2.0})
    r2 = BaseRanker._ranks_from_scores({"m": 2.0, "a": float("inf"), "z": float("nan")})
    assert r1 == r2 == ["a", "m", "z"]


def test_cosine_finite_inputs() -> None:
    # Parallel vectors -> +1; antiparallel -> -1.
    assert math.isclose(
        _cosine(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])), 1.0
    )
    assert math.isclose(
        _cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])), -1.0
    )


def test_cdscr_partial_corr_adjusted_zero_on_independent_data() -> None:
    # H3: the unadjusted multiple R^2 over p predictors is upward-biased; for
    # independent y and X (n=20, p=8) it returns ~0.46, reordering metrics by
    # descriptor count. The adjusted R^2 correctly reports ~0 association.
    from mriforge.core.metrics.meta_evaluation.rankers.cdscr import _partial_corr

    torch.manual_seed(0)
    n = 20
    y = torch.randn(n, dtype=torch.float64)
    x_design = torch.randn(n, 8, dtype=torch.float64)
    z_ctrl = torch.randn(n, 2, dtype=torch.float64)
    assert _partial_corr(y, x_design, z_ctrl) < 0.2
    # A genuine linear dependence on a column is still recovered (high).
    y_dep = x_design[:, 0] * 2.0 + 0.1 * torch.randn(n, dtype=torch.float64)
    assert _partial_corr(y_dep, x_design, z_ctrl) > 0.8


def test_cosine_nonfinite_returns_zero() -> None:
    # NaN/inf in either vector (e.g. a metric that produced a non-finite gradient)
    # must yield a finite, neutral 0.0 — not propagate NaN into the SFA score.
    assert (
        _cosine(torch.tensor([float("nan"), 1.0, 2.0]), torch.tensor([1.0, 2.0, 3.0]))
        == 0.0
    )
    assert _cosine(torch.tensor([1.0, 2.0]), torch.tensor([float("inf"), 2.0])) == 0.0
    assert _cosine(torch.tensor([float("nan")]), torch.tensor([float("nan")])) == 0.0


def test_cdscr_ranks_real_metric_above_constant(small_dataset, metric_set) -> None:
    result = CDSCRRanker().rank(metric_set, small_dataset)
    assert result.method == "CDSCR"
    assert "mse" in result.scores
    assert result.scores["mse"] > result.scores["constant"]


def test_lgdr_returns_finite_scores(small_dataset, metric_set) -> None:
    result = LGDRRanker().rank(metric_set, small_dataset)
    assert result.method == "LGDR"
    for v in result.scores.values():
        assert v == v  # not NaN
    # MSE should align with diffusion distance better than a noise metric does.
    assert result.scores["mse"] > result.scores["noise"]


def test_sfa_real_metric_aligned_with_score(small_dataset, metric_set) -> None:
    result = SFARanker().rank(metric_set, small_dataset)
    assert result.method == "SFA"
    # SFA computes signed cosine — should be > 0 for a real metric and
    # roughly 0 (within noise floor) for the constant metric.
    assert result.scores["mse"] > -0.5
    assert abs(result.scores["constant"]) < 0.5


def test_esd_real_metric_higher_ratio(small_dataset, metric_set) -> None:
    result = ESDRanker().rank(metric_set, small_dataset)
    assert result.method == "ESD"
    # MSE responds to lesion injection but is invariant to phase, etc.;
    # constant metric has 0 sensitivity_gain and 0 invariance_gap, so its
    # ratio is well-defined small (eps in numerator).
    assert result.scores["mse"] > result.scores["constant"]


def test_fspd_real_metric_loads_on_mode_one(small_dataset, metric_set) -> None:
    result = FSPDRanker().rank(metric_set, small_dataset)
    assert result.method == "FSPD"
    # MSE's response surface is monotone in severity -> should land on mode 1.
    assert result.scores["mse"] > result.scores["noise"]
    assert result.diagnostics["n_modes"] >= 1
