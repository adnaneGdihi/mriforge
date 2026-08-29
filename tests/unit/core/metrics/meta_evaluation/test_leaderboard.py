"""The leaderboard cube: tie groups, equal votes, and the stamped pool size."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.meta_evaluation.leaderboard import (  # noqa: E402
    DEFAULT_MIN_SAMPLES,
    CellKey,
    LeaderboardCell,
    LeaderboardCube,
    build_cell,
)
from mriforge.core.metrics.meta_evaluation.types import RankingResult  # noqa: E402


def _results(scores: dict[str, float], method: str = "cdscr") -> list[RankingResult]:
    ranks = sorted(scores, key=lambda m: -scores[m])
    return [RankingResult(method=method, scores=dict(scores), ranks=ranks)]


def _cell(
    scores: dict[str, float],
    *,
    n: int = 200,
    region: str = "full",
    artifact: str = "gaussian",
    contrast: str = "AXT2",
    pool_size: int = 10,
    excluded: dict[str, str] | None = None,
) -> LeaderboardCell:
    return build_cell(
        CellKey(artifact, region, contrast),
        _results(scores),
        n_samples=n,
        pool_size=pool_size,
        excluded=excluded,
    )


class TestTieGroups:
    """'Which 5 are best' invites a total order the data usually cannot support."""

    def test_well_separated_metrics_get_distinct_ranks(self) -> None:
        cell = _cell({"a": 3.0, "b": 1.0, "c": -1.0, "d": -3.0}, n=500)
        assert cell.fully_ordered
        assert [e.rank for e in cell.entries] == [1, 2, 3, 4]
        assert cell.n_tie_groups == 4

    def test_indistinguishable_metrics_share_a_rank(self) -> None:
        """Printing 1,2,3 when the evidence supports {1,2,3} is not a ranking -- it
        is a random number generator with good manners."""
        cell = _cell({"a": 1.0, "b": 1.0 + 1e-9, "c": 1.0 + 2e-9}, n=30)
        assert not cell.fully_ordered
        assert cell.n_tie_groups == 1
        assert {e.rank for e in cell.entries} == {1}

    def test_separable_from_next_is_reported_per_row(self) -> None:
        """The column that keeps the table honest: it says, in the table, that the
        next metric down is not demonstrably worse.

        Keyed by POSITION, not by metric name: a and b are separated by 1e-9, so
        which of them sorts first is exactly the coin flip under test. Asserting
        'a comes first' would be asserting the very thing we are calling undecidable.
        """
        cell = _cell({"a": 5.0, "b": 5.0 + 1e-9, "c": -5.0}, n=500)
        first, second, third = cell.entries

        assert {first.metric, second.metric} == {"a", "b"}
        assert third.metric == "c"

        assert not first.separable_from_next  # the near-tie: a coin flip
        assert second.separable_from_next  # the real gap down to c
        assert first.tie_group == second.tie_group
        assert third.tie_group != first.tie_group

    def test_a_tie_of_two_skips_the_next_rank(self) -> None:
        """Competition ranking: 1, 1, 3 -- not 1, 1, 2."""
        cell = _cell({"a": 5.0, "b": 5.0 + 1e-9, "c": -5.0}, n=500)
        assert [e.rank for e in cell.entries] == [1, 1, 3]

    def test_the_gap_threshold_is_stamped(self) -> None:
        """So a reader can see WHAT the separability claim was tested against."""
        assert _cell({"a": 1.0, "b": 0.0}, n=100).gap_threshold > 0


class TestPoolSizeIsStamped:
    def test_the_cell_records_how_many_metrics_were_eligible(self) -> None:
        """The denominator of every claim. The report says 'over the N eligible
        metrics', never 'over 164'."""
        cell = _cell({"a": 1.0, "b": 0.0}, pool_size=2, excluded={"ms_ssim": "too_small"})
        assert cell.pool_size == 2
        assert cell.excluded == {"ms_ssim": "too_small"}

    def test_excluded_metrics_are_not_ranked_last(self) -> None:
        """An all-NaN column ranked 'last' reads as 'this metric is bad at grey
        matter' when the truth is 'this metric has no meaning on grey matter'."""
        cell = _cell({"a": 1.0}, pool_size=1, excluded={"g_factor": "not_restrictable"})
        assert [e.metric for e in cell.entries] == ["a"]
        assert "g_factor" not in {e.metric for e in cell.entries}

    def test_every_row_carries_the_pool_size(self) -> None:
        rows = _cell({"a": 1.0, "b": 0.0}, pool_size=7).to_rows()
        assert all(r["pool_size"] == 7 for r in rows)


class TestUnderpoweredCells:
    def test_a_thin_cell_is_flagged(self) -> None:
        assert _cell({"a": 1.0, "b": 0.0}, n=DEFAULT_MIN_SAMPLES - 1).underpowered

    def test_a_thin_cell_is_reported_but_does_not_vote(self) -> None:
        """Reported, not dropped: silently discarding it hides that the design never
        had power there, which is itself a result. But letting noise outvote evidence
        would corrupt the consensus."""
        strong = _cell({"a": 3.0, "b": 1.0}, n=500, region="full")
        thin = _cell({"a": 1.0, "b": 3.0}, n=5, region="path:lesion_any")
        cube = LeaderboardCube((strong, thin))

        assert len(cube) == 2  # both reported
        assert len(cube.underpowered) == 1

        result = cube.consensus()
        assert result.n_voters == 1  # only the powered cell voted
        assert len(result.excluded_voters) == 1
        assert "min_samples" in result.excluded_voters[0][1]


class TestEqualCellVotes:
    def test_equal_is_the_default(self) -> None:
        cube = LeaderboardCube((_cell({"a": 2.0, "b": 1.0}),))
        assert cube.consensus().vote_weight == "equal"

    def test_size_weighting_raises_rather_than_silently_obliging(self) -> None:
        """Size-weighting lets the ~100k-px `full` cell drown the ~1.6k-px lesion
        cell -- and the divergence between exactly those two IS the experiment.
        Weighting by n would answer the question by assuming the answer."""
        cube = LeaderboardCube((_cell({"a": 2.0, "b": 1.0}),))
        with pytest.raises(ValueError, match="see through"):
            cube.consensus(vote_weight="by_size")

    def test_a_big_cell_does_not_outvote_a_small_one(self) -> None:
        """Two cells, opposite orders, both adequately powered -> a genuine tie,
        not a win for whichever region has more pixels."""
        big = _cell({"a": 3.0, "b": 1.0}, n=1000, region="full")
        small = _cell({"a": 1.0, "b": 3.0}, n=100, region="path:lesion_any")
        result = LeaderboardCube((big, small)).consensus()
        assert result.n_voters == 2  # one cell, one vote


class TestMarginals:
    @pytest.fixture
    def cube(self) -> LeaderboardCube:
        return LeaderboardCube(
            (
                _cell({"a": 3.0, "b": 1.0}, region="full", contrast="AXT2"),
                _cell({"a": 1.0, "b": 3.0}, region="path:lesion_any", contrast="AXT2"),
                _cell({"a": 2.0, "b": 1.0}, region="full", contrast="AXFLAIR"),
            )
        )

    def test_slicing_selects_the_right_cells(self, cube) -> None:
        assert len(cube.slice(region="full")) == 2
        assert len(cube.slice(contrast="AXFLAIR")) == 1
        assert len(cube.slice(region="full", contrast="AXT2")) == 1

    def test_by_region_collapses_artifacts_and_contrasts(self, cube) -> None:
        by = cube.by_region()
        assert set(by) == {"full", "path:lesion_any"}
        assert by["full"].n_voters == 2

    def test_by_anatomy_collapses_artifacts_and_regions(self, cube) -> None:
        by = cube.by_anatomy()
        assert set(by) == {"AXT2", "AXFLAIR"}
        assert by["AXT2"].n_voters == 2

    def test_by_artifact_collapses_regions_and_contrasts(self, cube) -> None:
        assert set(cube.by_artifact()) == {"gaussian"}


class TestDefensibilitySemanticsStaysSeparate:
    def test_the_cell_stamps_breadth(self) -> None:
        """aggregate() writes 'breadth' into defensibility_flags..."""
        assert _cell({"a": 1.0, "b": 0.0}).defensibility_semantics == "breadth"

    def test_the_consensus_stamps_condorcet(self) -> None:
        """...while kemeny_consensus() writes 'condorcet' into the SAME field name.
        Incompatible events -- so they are rendered in separate columns and never
        compared."""
        cube = LeaderboardCube((_cell({"a": 2.0, "b": 1.0}),))
        assert cube.consensus().to_dict()["defensibility_semantics"] == "condorcet"


class TestEmptyPool:
    def test_a_cell_with_no_eligible_metrics_yields_no_entries(self) -> None:
        cell = build_cell(
            CellKey("gaussian", "tissue:gm", "AXT2"),
            _results({}),
            n_samples=100,
            pool_size=0,
        )
        assert cell.entries == ()

    def test_an_empty_cell_does_not_vote(self) -> None:
        cell = build_cell(
            CellKey("gaussian", "tissue:gm", "AXT2"),
            _results({}),
            n_samples=100,
            pool_size=0,
        )
        result = LeaderboardCube((cell,)).consensus()
        assert result.n_voters == 0
        assert result.excluded_voters[0][1] == "empty metric pool"
