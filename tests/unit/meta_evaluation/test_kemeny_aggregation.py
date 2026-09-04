"""Tests for the L3⁺ Kemeny-median (Condorcet-consistent) aggregator.

These pin the executable counterpart of the Lean theorems in
``docs/lean/Sim2Rank/Sim2Rank/Kemeny.lean``: a Condorcet / Pareto winner is
ranked first (``kemeny_optimal_ranks_*``), strictly stronger than the
z-aggregator's Pareto-only guarantee.
"""

from __future__ import annotations

from spectramr.core.metrics.meta_evaluation.aggregation import (
    KemenyConfig,
    kemeny_consensus,
)
from spectramr.core.metrics.meta_evaluation.types import RankingResult


def _axis(method: str, order: list[str]) -> RankingResult:
    """A RankingResult whose scores encode the given best-to-worst order."""
    n = len(order)
    scores = {m: float(n - 1 - i) for i, m in enumerate(order)}
    return RankingResult(method=method, scores=scores, ranks=list(order))


def test_pareto_winner_ranked_first() -> None:
    # 'a' is first on every axis -> Condorcet winner -> ranked first (Lean capstone).
    results = [
        _axis("ax1", ["a", "b", "c"]),
        _axis("ax2", ["a", "c", "b"]),
        _axis("ax3", ["a", "b", "c"]),
    ]
    agg = kemeny_consensus(results)
    assert agg.final_ranks[0] == "a"
    assert agg.defensibility_flags["a"] is True


def test_condorcet_winner_ranked_first_without_unanimity() -> None:
    # 'a' loses one axis but wins the pairwise majority against everyone.
    results = [
        _axis("ax1", ["a", "b", "c"]),
        _axis("ax2", ["a", "b", "c"]),
        _axis("ax3", ["b", "c", "a"]),  # a last here
    ]
    agg = kemeny_consensus(results)
    # a beats b 2-1 and c 2-1 -> Condorcet winner -> first.
    assert agg.final_ranks[0] == "a"
    assert agg.defensibility_flags["a"] is True


def test_majority_consensus_full_order() -> None:
    # Clean unanimous order a > b > c -> consensus reproduces it.
    results = [_axis(f"ax{i}", ["a", "b", "c"]) for i in range(5)]
    agg = kemeny_consensus(results)
    assert agg.final_ranks == ["a", "b", "c"]
    # final_scores are position-descending so argsort reproduces the order.
    assert sorted(agg.final_scores, key=lambda m: -agg.final_scores[m]) == ["a", "b", "c"]


def test_cycle_has_no_condorcet_winner() -> None:
    # Condorcet cycle a>b, b>c, c>a -> no metric beats all others.
    results = [
        _axis("ax1", ["a", "b", "c"]),
        _axis("ax2", ["b", "c", "a"]),
        _axis("ax3", ["c", "a", "b"]),
    ]
    agg = kemeny_consensus(results)
    assert not any(agg.defensibility_flags.values())
    # Kemeny still returns a valid full order over all metrics.
    assert sorted(agg.final_ranks) == ["a", "b", "c"]


def test_local_search_matches_exact_on_small_pool() -> None:
    # Forcing local search (max_exact=2) must agree with exact on a clear order.
    results = [_axis(f"ax{i}", ["a", "b", "c", "d"]) for i in range(3)]
    exact = kemeny_consensus(results, KemenyConfig(max_exact=8))
    local = kemeny_consensus(results, KemenyConfig(max_exact=2))
    assert exact.final_ranks == local.final_ranks == ["a", "b", "c", "d"]


def test_empty_results() -> None:
    agg = kemeny_consensus([])
    assert agg.final_ranks == []
    assert agg.final_scores == {}


def test_weighted_axes_break_ties() -> None:
    # A heavily-weighted dissenting axis can flip the consensus.
    results = [
        _axis("strong", ["b", "a"]),
        _axis("weak1", ["a", "b"]),
    ]
    agg = kemeny_consensus(results, KemenyConfig(weights={"strong": 3.0, "weak1": 1.0}))
    assert agg.final_ranks[0] == "b"


def test_kemeny_stamps_condorcet_semantics() -> None:
    """Kemeny labels its defensibility flag 'condorcet' so it isn't compared with
    the z-aggregator's 'breadth' flag (critique A2)."""
    results = [_axis(f"ax{i}", ["a", "b", "c"]) for i in range(3)]
    agg = kemeny_consensus(results)
    assert agg.defensibility_semantics == "condorcet"


def test_single_metric_pool_is_not_a_spurious_condorcet_winner() -> None:
    """A one-metric pool has no opponents, so ``all(<empty>)`` must NOT certify it
    as a Condorcet winner (critique A6)."""
    results = [_axis("ax1", ["only"]), _axis("ax2", ["only"])]
    agg = kemeny_consensus(results)
    assert agg.final_ranks == ["only"]
    assert agg.defensibility_flags["only"] is False
