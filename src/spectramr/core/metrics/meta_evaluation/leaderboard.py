"""The region-conditional leaderboard cube: (artifact x region x contrast).

This module **reuses** the existing machinery and invents no new scorer:

* :func:`~spectramr.core.metrics.meta_evaluation.aggregation.aggregate` -- the
  z-weighted cross-ranker combiner, per cell.
* :func:`~spectramr.core.metrics.meta_evaluation.powered.certify_partial_order` --
  the Hoeffding-bounded pairwise gate, which is what turns "top 5" into an honest
  claim.
* :func:`~spectramr.core.metrics.meta_evaluation.aggregation.kemeny_consensus` --
  the Condorcet-consistent consensus, with each cell as one voter.

Three things it adds, each of which exists to stop a specific lie.

Tie groups
----------
"Which 5 metrics are best" invites a total order that the data usually cannot
support. ``certify_partial_order`` already knows which pairwise gaps clear the
powered threshold at ``(N, alpha)``; adjacent top-K entries whose pair is
**not** certified are collapsed into a **tie group** and rendered at a shared
rank. A table that prints 1,2,3,4,5 when the evidence supports 1,{2,3},{4,5} is
not a ranking -- it is a random number generator with good manners.

Equal cell votes
----------------
``cell_vote_weight="equal"`` is the default. Size-weighting would let the big
``full`` cell (100k px) drown the small ``path:lesion_any`` cell (1.6k px) -- and
the divergence between those two *is the entire experiment*. Weighting by n would
answer the question by assuming the answer.

Pool size is stamped, never implied
-----------------------------------
Ineligible metrics are **excluded from the cell's pool**, not carried as all-NaN
columns. An all-NaN column ranked "last" reads as *"this metric is bad at grey
matter"* when the truth is *"this metric has no meaning on grey matter"*. Those
are different claims. Every cell therefore carries ``pool_size`` and
``excluded``, and the report says "leaderboard over the N eligible metrics" --
never "over 164".
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from spectramr.core.metrics.meta_evaluation.aggregation import (
    AggregatorConfig,
    KemenyConfig,
    aggregate,
    kemeny_consensus,
)
from spectramr.core.metrics.meta_evaluation.powered import certify_partial_order
from spectramr.core.metrics.meta_evaluation.types import (
    AggregateRankingResult,
    RankingResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "CellKey",
    "ConsensusResult",
    "LeaderboardCell",
    "LeaderboardCube",
    "LeaderboardEntry",
    "build_cell",
]

DEFAULT_MIN_SAMPLES = 30
"""Below this, a cell's ranking is reported but excluded from the consensus vote.

Not dropped -- *reported*. Silently discarding a thin cell hides the fact that the
design never had power there, which is itself a result (usually: this region is too
rare in this cohort). But letting it vote would let noise outvote evidence.
"""


@dataclass(frozen=True, slots=True)
class CellKey:
    """One (artifact, region, contrast) cell of the cube."""

    artifact: str
    region: str
    contrast: str

    def __str__(self) -> str:
        return f"{self.artifact}|{self.region}|{self.contrast}"


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """One row of a cell's top-K."""

    rank: int
    """Shared across a tie group -- so ranks may repeat, and may skip."""

    metric: str
    score: float
    tie_group: int
    separable_from_next: bool
    """False when the gap to the next entry does not clear the powered threshold.

    This is the column that keeps the table honest: it says, in the table itself,
    that the next metric down is not demonstrably worse.
    """

    def as_row(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "metric": self.metric,
            "score": round(self.score, 6),
            "tie_group": self.tie_group,
            "separable_from_next": self.separable_from_next,
        }


@dataclass(frozen=True, slots=True)
class LeaderboardCell:
    """A ranked, power-gated leaderboard for one cell."""

    key: CellKey
    entries: tuple[LeaderboardEntry, ...]
    n_samples: int
    pool_size: int
    """How many metrics were ELIGIBLE here. The denominator of every claim."""

    excluded: Mapping[str, str] = field(default_factory=dict)
    """``metric -> reason``. Ineligible metrics, with why. Never silently dropped."""

    gap_threshold: float = 0.0
    defensibility_semantics: str = "breadth"
    """"breadth" (aggregate) or "condorcet" (kemeny). The SAME field name carries
    incompatible meanings across the two combiners, so it is stamped and the two are
    rendered in SEPARATE columns -- never compared."""

    @property
    def underpowered(self) -> bool:
        return self.n_samples < DEFAULT_MIN_SAMPLES

    @property
    def n_tie_groups(self) -> int:
        return len({e.tie_group for e in self.entries})

    @property
    def fully_ordered(self) -> bool:
        """True when every adjacent pair in the top-K is certified."""
        return self.n_tie_groups == len(self.entries)

    def top(self, k: int = 5) -> tuple[LeaderboardEntry, ...]:
        return self.entries[:k]

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "artifact": self.key.artifact,
                "region": self.key.region,
                "contrast": self.key.contrast,
                "n": self.n_samples,
                "pool_size": self.pool_size,
                "underpowered": self.underpowered,
                "gap_threshold": round(self.gap_threshold, 6),
                "defensibility_semantics": self.defensibility_semantics,
                **e.as_row(),
            }
            for e in self.entries
        ]


def _tie_groups(
    ranks: Sequence[str],
    scores: Mapping[str, float],
    n_samples: int,
    alpha: float,
) -> tuple[list[LeaderboardEntry], float]:
    """Collapse adjacent, non-separable metrics into shared ranks.

    ``certify_partial_order`` returns the pairs whose gap clears the powered
    threshold. Any adjacent pair NOT in that set is statistically indistinguishable
    at ``(N, alpha)``, and forcing them apart would be inventing precision.
    """
    order = certify_partial_order({m: scores[m] for m in ranks}, n_samples, alpha)
    certified = set(order.certified)

    entries: list[LeaderboardEntry] = []
    group = 0
    rank = 1
    for i, metric in enumerate(ranks):
        nxt = ranks[i + 1] if i + 1 < len(ranks) else None
        separable = nxt is not None and (metric, nxt) in certified

        entries.append(
            LeaderboardEntry(
                rank=rank,
                metric=metric,
                score=float(scores[metric]),
                tie_group=group,
                separable_from_next=bool(separable),
            )
        )
        if nxt is not None and separable:
            group += 1
            rank = i + 2  # competition ranking: a tie of 2 skips the next rank

    return entries, order.gap_threshold


def build_cell(
    key: CellKey,
    results: list[RankingResult],
    *,
    n_samples: int,
    pool_size: int,
    excluded: Mapping[str, str] | None = None,
    top_k: int = 5,
    alpha: float = 0.05,
    config: AggregatorConfig | None = None,
) -> LeaderboardCell:
    """Aggregate one cell's rankers, then power-gate the resulting order."""
    agg: AggregateRankingResult = aggregate(results, config)
    ranks = agg.final_ranks[: max(top_k, 1)]

    if not ranks:
        return LeaderboardCell(
            key=key,
            entries=(),
            n_samples=n_samples,
            pool_size=pool_size,
            excluded=dict(excluded or {}),
            defensibility_semantics=agg.defensibility_semantics,
        )

    entries, threshold = _tie_groups(ranks, agg.final_scores, n_samples, alpha)
    return LeaderboardCell(
        key=key,
        entries=tuple(entries),
        n_samples=n_samples,
        pool_size=pool_size,
        excluded=dict(excluded or {}),
        gap_threshold=threshold,
        defensibility_semantics=agg.defensibility_semantics,
    )


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """A Kemeny consensus over cells, plus who was allowed to vote."""

    ranks: tuple[str, ...]
    scores: Mapping[str, float]
    condorcet_flags: Mapping[str, bool]
    n_voters: int
    excluded_voters: tuple[tuple[str, str], ...]
    """``(cell, reason)`` -- underpowered cells that were reported but did not vote."""

    vote_weight: str = "equal"

    def to_dict(self) -> dict[str, object]:
        return {
            "vote_weight": self.vote_weight,
            "n_voters": self.n_voters,
            "excluded_voters": [{"cell": c, "reason": r} for c, r in self.excluded_voters],
            # "condorcet", NOT "breadth". Kemeny's defensibility flag means a strict
            # Condorcet winner; aggregate()'s means "positive on enough axes". Same
            # field name, incompatible events -- so they get separate columns.
            "defensibility_semantics": "condorcet",
            "ranks": list(self.ranks),
            "condorcet_winners": sorted(m for m, flag in self.condorcet_flags.items() if flag),
        }


@dataclass(frozen=True, slots=True)
class LeaderboardCube:
    """Every cell, plus the marginals and the consensus."""

    cells: tuple[LeaderboardCell, ...]

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def underpowered(self) -> tuple[LeaderboardCell, ...]:
        return tuple(c for c in self.cells if c.underpowered)

    def slice(
        self,
        *,
        artifact: str | None = None,
        region: str | None = None,
        contrast: str | None = None,
    ) -> tuple[LeaderboardCell, ...]:
        return tuple(
            c
            for c in self.cells
            if (artifact is None or c.key.artifact == artifact)
            and (region is None or c.key.region == region)
            and (contrast is None or c.key.contrast == contrast)
        )

    def consensus(
        self,
        cells: Sequence[LeaderboardCell] | None = None,
        *,
        vote_weight: str = "equal",
        config: KemenyConfig | None = None,
    ) -> ConsensusResult:
        """Kemeny consensus with **one vote per cell**.

        ``vote_weight="equal"`` is the only implemented policy, and deliberately so.
        Size-weighting would let the ~100k-pixel ``full`` cell drown the ~1.6k-pixel
        ``path:lesion_any`` cell -- and the divergence between exactly those two is
        the experiment. Weighting by n would answer the question by assuming the
        answer, so an explicit request for it raises rather than silently obliging.
        """
        if vote_weight != "equal":
            raise ValueError(
                f"cell_vote_weight={vote_weight!r} is not supported. Only 'equal' is: "
                "size-weighting lets the big whole-slice cell outvote the small "
                "lesion cell, which is precisely the effect the region axis exists to "
                "see through."
            )

        pool = list(cells if cells is not None else self.cells)

        # Underpowered cells are REPORTED but do not vote -- letting noise outvote
        # evidence would corrupt the consensus, and silently dropping them would
        # hide that the design had no power there.
        voters, excluded = [], []
        for c in pool:
            if c.underpowered:
                excluded.append(
                    (str(c.key), f"n={c.n_samples} < min_samples={DEFAULT_MIN_SAMPLES}")
                )
                continue
            if not c.entries:
                excluded.append((str(c.key), "empty metric pool"))
                continue
            voters.append(
                RankingResult(
                    method=str(c.key),
                    scores={e.metric: e.score for e in c.entries},
                    ranks=[e.metric for e in c.entries],
                )
            )

        if excluded:
            logger.warning(
                "consensus(): %d/%d cells excluded from the vote: %s",
                len(excluded),
                len(pool),
                [c for c, _ in excluded],
            )

        if not voters:
            return ConsensusResult(
                ranks=(),
                scores={},
                condorcet_flags={},
                n_voters=0,
                excluded_voters=tuple(excluded),
                vote_weight=vote_weight,
            )

        agg = kemeny_consensus(voters, config)
        return ConsensusResult(
            ranks=tuple(agg.final_ranks),
            scores=dict(agg.final_scores),
            condorcet_flags=dict(agg.defensibility_flags),
            n_voters=len(voters),
            excluded_voters=tuple(excluded),
            vote_weight=vote_weight,
        )

    def by_region(self, **kw: object) -> dict[str, ConsensusResult]:
        """Marginal over artifacts and contrasts: one consensus per region."""
        regions = sorted({c.key.region for c in self.cells})
        return {
            r: self.consensus(self.slice(region=r), **kw)  # type: ignore[arg-type]
            for r in regions
        }

    def by_anatomy(self, **kw: object) -> dict[str, ConsensusResult]:
        """Marginal over artifacts and regions: one consensus per contrast."""
        contrasts = sorted({c.key.contrast for c in self.cells})
        return {
            k: self.consensus(self.slice(contrast=k), **kw)  # type: ignore[arg-type]
            for k in contrasts
        }

    def by_artifact(self, **kw: object) -> dict[str, ConsensusResult]:
        """Marginal over regions and contrasts: one consensus per artifact."""
        artifacts = sorted({c.key.artifact for c in self.cells})
        return {
            a: self.consensus(self.slice(artifact=a), **kw)  # type: ignore[arg-type]
            for a in artifacts
        }

    def to_rows(self) -> list[dict[str, object]]:
        return [row for c in self.cells for row in c.to_rows()]
