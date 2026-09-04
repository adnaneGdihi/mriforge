"""Tests for the shared ranker guards (tie-safe ranks + degeneracy detection)."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.ranking_guards import (
    DegenerateRankingError,
    assert_not_degenerate,
    distinct_fraction,
    order_from_scores,
    ranks_from_scores,
)


class TestDistinctFraction:
    def test_all_distinct(self) -> None:
        assert distinct_fraction([1.0, 2.0, 3.0, 4.0]) == 1.0

    def test_all_tied(self) -> None:
        assert distinct_fraction([1.0] * 8) == pytest.approx(1 / 8)

    def test_float_dust_is_not_discrimination(self) -> None:
        # Values differing below the rounding floor must count as one.
        assert distinct_fraction([1.0, 1.0 + 1e-15, 1.0 + 2e-15]) == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        assert distinct_fraction([]) == 0.0


class TestAssertNotDegenerate:
    def test_healthy_pool_passes(self) -> None:
        assert_not_degenerate(np.linspace(0, 1, 10), name="adr")

    def test_all_tied_raises(self) -> None:
        with pytest.raises(DegenerateRankingError, match="carries no information"):
            assert_not_degenerate(np.ones(10), name="adr")

    def test_regression_calibration_leak_shape(self) -> None:
        """The #236 signature: every monotone metric tied at 1.0, a couple of stragglers.

        This is the exact vector the calibrated-ADR pipeline produced. It must raise —
        before this guard existed the leaderboard happily printed a top-5 from it.
        """
        scores = np.ones(40)
        scores[[7, 23]] = [0.42, 0.31]
        with pytest.raises(DegenerateRankingError, match=r"3 distinct scores across 40"):
            assert_not_degenerate(scores, name="adr")

    def test_tiny_pool_is_exempt(self) -> None:
        assert_not_degenerate([1.0, 1.0], name="adr")

    def test_threshold_is_configurable(self) -> None:
        scores = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])  # 4/6 distinct
        assert_not_degenerate(scores, name="x", min_distinct_frac=0.5)
        with pytest.raises(DegenerateRankingError):
            assert_not_degenerate(scores, name="x", min_distinct_frac=0.9)


class TestNonsignalFloor:
    """Per-axis ADR: a floor tie is not the ceiling saturation the guard was built for.

    On a low-pass axis (e.g. gibbs) most HF-sensitive metrics legitimately clamp to the
    ADR floor 0.0 (``max(0, -s*rho)``); the minority that respond still rank fine. The
    whole-pool distinct-fraction reads that as degeneracy and blocks the run — the
    ``DegenerateRankingError: Ranker 'ADR[gibbs]' produced 56 distinct scores across 146
    metrics (38.4%)`` crash. ``nonsignal_floor`` drops the floor/non-finite metrics and
    asserts discriminability on the signal-carriers only.
    """

    @staticmethod
    def _gibbs_like(n_floor: int = 90, n_live: int = 56, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.concatenate([np.zeros(n_floor), rng.uniform(0.1, 0.9, n_live)])

    def test_floor_tie_passes_when_floor_excluded(self) -> None:
        # The exact crash vector: 39% distinct over the whole pool → would raise;
        # the signal-carriers alone are fully distinct → the ranking is meaningful.
        adr = self._gibbs_like()
        assert distinct_fraction(adr) < 0.5
        assert_not_degenerate(adr, name="ADR[gibbs]", nonsignal_floor=0.0)

    def test_same_vector_still_raises_without_the_opt_in(self) -> None:
        # Default (undirected) behaviour is unchanged — the guard still fires.
        with pytest.raises(DegenerateRankingError, match="carries no information"):
            assert_not_degenerate(self._gibbs_like(), name="ADR[gibbs]")

    def test_ceiling_saturation_still_raises_with_floor_set(self) -> None:
        """#236 must survive: a collapse to a constant *non-floor* value (1.0) is not a
        floor tie — that value stays in the live pool and is still caught."""
        scores = np.ones(40)
        scores[[7, 23]] = [0.42, 0.31]
        with pytest.raises(DegenerateRankingError, match="#236"):
            assert_not_degenerate(scores, name="ADR[axis]", nonsignal_floor=0.0)

    def test_total_collapse_to_floor_raises(self) -> None:
        """Nothing responds to the axis (every metric floored) → real degeneracy."""
        with pytest.raises(DegenerateRankingError, match="ranking is empty"):
            assert_not_degenerate(np.zeros(50), name="ADR[dead]", nonsignal_floor=0.0)

    def test_all_nonfinite_collapse_raises(self) -> None:
        """Every trajectory was NaN → the live pool is empty → raise, don't pass."""
        with pytest.raises(DegenerateRankingError, match="ranking is empty"):
            assert_not_degenerate(np.full(50, np.nan), name="ADR[nan]", nonsignal_floor=0.0)

    def test_degenerate_live_population_raises(self) -> None:
        """The survivors themselves collapse into one tie → the #236 pathology, caught."""
        adr = np.concatenate([np.zeros(80), np.full(60, 0.7)])  # 60 tied survivors
        with pytest.raises(DegenerateRankingError, match="signal-carrying"):
            assert_not_degenerate(adr, name="ADR[sat]", nonsignal_floor=0.0)

    def test_nonfinite_counts_as_nonsignal_alongside_floor(self) -> None:
        # NaN metrics (never-measured) join the floored ones as non-signal; the finite
        # signal-carriers still separate the pool.
        adr = np.concatenate([np.zeros(40), np.full(40, np.nan), np.linspace(0.1, 0.9, 20)])
        assert_not_degenerate(adr, name="ADR[mix]", nonsignal_floor=0.0)

    def test_floor_value_is_configurable(self) -> None:
        # The sentinel need not be 0.0 — a directed ranker clamped elsewhere is covered.
        adr = np.concatenate([np.full(90, -1.0), np.linspace(0.1, 0.9, 56)])
        assert_not_degenerate(adr, name="ADR[shift]", nonsignal_floor=-1.0)


class TestRanksFromScores:
    def test_descending_and_one_indexed(self) -> None:
        r = ranks_from_scores([0.1, 0.9, 0.5])
        assert list(r) == [3.0, 1.0, 2.0]

    def test_ties_share_a_midrank_and_are_not_truncated(self) -> None:
        """Regression for #239: rankdata(average).astype(int64) floored 1.5 -> 1."""
        r = ranks_from_scores([0.9, 0.9, 0.5, 0.1])
        assert list(r) == [1.5, 1.5, 3.0, 4.0]
        assert r.dtype == np.float64

    def test_no_positional_tie_break(self) -> None:
        """Regression for #243: a fully tied pool must NOT be ordered by index."""
        r = ranks_from_scores(np.ones(5))
        assert np.allclose(r, 3.0), "every tied metric must share the same mid-rank"

    def test_posinf_ranks_first(self) -> None:
        """Regression for #240: +inf is the ideal SCVR case, not the worst."""
        r = ranks_from_scores([np.inf, 10.0, 1.0])
        assert r[0] == 1.0

    def test_nan_ranks_last(self) -> None:
        """Regression for #238/#251: a crashed metric must never surface at the top."""
        r = ranks_from_scores([0.9, np.nan, 0.5, 0.7])
        assert r[1] == 4.0

    def test_neginf_ranks_last(self) -> None:
        r = ranks_from_scores([0.9, -np.inf, 0.5])
        assert r[1] == 3.0

    def test_crashed_ties_broken_by_name_not_position(self) -> None:
        r1 = ranks_from_scores([np.nan, np.nan, 1.0], names=["zeta", "alpha", "ok"])
        r2 = ranks_from_scores([np.nan, np.nan, 1.0], names=["zeta", "alpha", "ok"])
        assert list(r1) == list(r2)
        # 'alpha' sorts before 'zeta', so it takes the better of the two trailing ranks.
        assert r1[1] < r1[0]

    def test_empty(self) -> None:
        assert ranks_from_scores([]).size == 0


class TestOrderFromScores:
    def test_best_first(self) -> None:
        assert order_from_scores({"a": 0.1, "b": 0.9, "c": 0.5}) == ["b", "c", "a"]

    def test_ties_broken_by_name_not_insertion_order(self) -> None:
        a = order_from_scores({"z": 1.0, "a": 1.0, "m": 1.0})
        b = order_from_scores({"m": 1.0, "z": 1.0, "a": 1.0})
        assert a == b == ["a", "m", "z"]

    def test_posinf_first_nan_last(self) -> None:
        out = order_from_scores({"good": 1.0, "ideal": np.inf, "crashed": np.nan})
        assert out[0] == "ideal"
        assert out[-1] == "crashed"
