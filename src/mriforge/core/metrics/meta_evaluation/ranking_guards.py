"""Tie-safe ranking + degeneracy detection shared by every sim2rank ranker.

Two failure modes recur across the meta-evaluation stack and are fixed here once:

**Positional tie-breaks.** ``np.argsort(scores)[::-1]`` (and ``argsort().argsort()``)
resolve ties by array index. A ranker whose scores collapse into one large tie block
therefore emits a ranking that is really the *registry order*, and any statistic
computed on it (Kendall's W across folds, an inter-method rho matrix, an "oracle
ranks #1" assertion) reports agreement that is only the determinism of ``argsort``.
:func:`ranks_from_scores` uses mid-ranks instead, so tied metrics share a rank and a
downstream correlation sees the tie for what it is.

**Silent degeneracy.** A ranker that assigns nearly every metric the same score has
failed, but nothing in the pipeline noticed: the leaderboard still printed a top-5.
:func:`assert_not_degenerate` turns that into a loud error. It is the single check
that catches an entire class of bugs (calibration leakage, rank-based statistics that
saturate at 1.0 on any monotone input, entropy scores that max out on noise).

Non-finite scores are *not* interchangeable. ``+inf`` is a legitimate best case for a
ratio statistic (e.g. SCVR with zero content nuisance), while ``NaN``/``-inf`` mean the
metric crashed and carries no signal. :func:`ranks_from_scores` ranks ``+inf`` first and
``NaN``/``-inf`` last, tie-broken by name so the order is reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


class DegenerateRankingError(ValueError):
    """A ranker produced too few distinct scores for its ranking to mean anything."""


def _as_array(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(
        list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float64
    )


def distinct_fraction(scores: Sequence[float] | np.ndarray, *, decimals: int = 9) -> float:
    """Fraction of the pool that carries a distinct score, in ``[0, 1]``.

    ``1.0`` means every metric is separated; ``1/M`` means the ranker returned a
    constant and the ranking is pure tie-break noise. Rounded to ``decimals`` so
    float dust does not masquerade as discrimination.
    """
    arr = _as_array(scores)
    if arr.size == 0:
        return 0.0
    return float(len(np.unique(np.round(arr, decimals))) / arr.size)


def assert_not_degenerate(
    scores: Sequence[float] | np.ndarray,
    *,
    name: str,
    min_distinct_frac: float = 0.5,
    decimals: int = 9,
    nonsignal_floor: float | None = None,
) -> None:
    """Raise :class:`DegenerateRankingError` when a ranker has collapsed into ties.

    Args:
        scores: The ranker's score vector.
        name: Ranker name, quoted in the error message.
        min_distinct_frac: Minimum fraction of distinct scores required. The default
            0.5 tolerates a genuinely tied pool half (clone metrics do exist) while
            still failing loudly on the pathologies this guard exists for.
        decimals: Rounding applied before counting distinct values.
        nonsignal_floor: When set, scores equal to this value (and non-finite scores)
            are treated as "no signal on this axis" and dropped from the
            discriminability population; the distinct-fraction floor is then applied to
            the *signal-carrying* survivors only. Use it for a **directed** ranker
            (ADR: ``max(0, -s*rho)``) scored **per axis**, where a metric that does not
            respond to that axis legitimately clamps to the floor. A *floor* tie leaves
            the leaderboard top correctly ordered, so it is not the ceiling saturation
            (#236) this guard exists for — a collapse to a constant *non-floor* value
            keeps that value in the live pool and is still caught. A pool that collapses
            *entirely* to the floor/non-finite (nothing responds to the axis) still
            raises. Leave ``None`` (default) for an undirected ranker to keep the
            whole-pool check.

    Raises:
        DegenerateRankingError: If the (surviving) distinct fraction is below the floor,
            or — with ``nonsignal_floor`` set — no metric carries signal at all. A pool
            of fewer than 3 metrics is exempt — there is nothing to be degenerate about.
    """
    arr = _as_array(scores)
    if arr.size < 3:
        return

    if nonsignal_floor is None:
        frac = distinct_fraction(arr, decimals=decimals)
        if frac < min_distinct_frac:
            n_distinct = round(frac * arr.size)
            raise DegenerateRankingError(
                f"Ranker {name!r} produced {n_distinct} distinct scores across "
                f"{arr.size} metrics ({frac:.1%} < {min_distinct_frac:.0%} required). "
                "The ranking carries no information — ties would be resolved by pool "
                "order, not quality. This usually means the statistic saturated (e.g. "
                "scored on a calibrated trajectory, or a rank-based statistic on "
                "monotone input)."
            )
        return

    # Directed per-axis path: the ADR clamp floor (and non-finite) marks a metric that
    # carries no signal on THIS axis. Dropping it distinguishes a benign floor tie
    # (survivors still rank fine at the top) from the ceiling saturation this guard was
    # built for (#236) — a collapse to a constant *non-floor* value leaves that value
    # in the live pool, so it is still caught.
    finite = np.isfinite(arr)
    at_floor = finite & (np.round(arr, decimals) == round(float(nonsignal_floor), decimals))
    live = arr[finite & ~at_floor]
    if live.size == 0:
        raise DegenerateRankingError(
            f"Ranker {name!r}: all {arr.size} metrics sit at the non-signal floor "
            f"({nonsignal_floor}) or are non-finite — no metric responds to this axis, "
            "so the ranking is empty. That is a real degeneracy (every trajectory was "
            "NaN, or every metric scored the wrong direction), not a benign floor tie."
        )
    frac = distinct_fraction(live, decimals=decimals)
    if frac < min_distinct_frac:
        n_distinct = round(frac * live.size)
        raise DegenerateRankingError(
            f"Ranker {name!r}: among the {live.size} signal-carrying metrics (of "
            f"{arr.size}; the other {arr.size - live.size} sit at the {nonsignal_floor} "
            f"floor or are non-finite), only {n_distinct} distinct scores "
            f"({frac:.1%} < {min_distinct_frac:.0%} required). The live population "
            "itself has collapsed into ties — the saturation this guard exists for "
            "(#236), not a benign floor tie."
        )


def ranks_from_scores(
    scores: Sequence[float] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """1-indexed mid-ranks, best (highest score) first. Ties share a rank.

    ``+inf`` outranks every finite score (a legitimate best case). ``NaN`` and
    ``-inf`` rank last — a crashed metric must never surface at the top — and are
    tie-broken by ``names`` when supplied, so the ordering does not depend on dict
    insertion order.

    Returns:
        ``(M,)`` float array of mid-ranks (``1.0`` = best). Ties yield half-integers;
        the ranks are deliberately **not** cast to int, since truncating ``1.5 → 1``
        silently duplicates rank 1 and drops rank 2.
    """
    from scipy.stats import rankdata

    arr = _as_array(scores)
    n = arr.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    finite = np.isfinite(arr)
    is_best = np.isposinf(arr)
    is_worst = ~finite & ~is_best  # NaN and -inf

    # Sort key: rank descending on score, so negate. +inf -> below every finite key,
    # NaN/-inf -> above every finite key (i.e. worst).
    key = np.empty(n, dtype=np.float64)
    if finite.any():
        lo, hi = float(arr[finite].min()), float(arr[finite].max())
    else:
        lo = hi = 0.0
    span = max(hi - lo, 1.0)
    key[finite] = -arr[finite]
    key[is_best] = -(hi + span)
    key[is_worst] = -(lo - span)

    # Break crashed-metric ties by name so the order is reproducible.
    if names is not None and is_worst.any():
        order = {nm: i for i, nm in enumerate(sorted(np.asarray(names)[is_worst]))}
        bump = np.zeros(n, dtype=np.float64)
        eps = span * 1e-9
        for i in np.flatnonzero(is_worst):
            bump[i] = order[str(np.asarray(names)[i])] * eps
        key = key + bump

    return np.asarray(rankdata(key, method="average"), dtype=np.float64)


def order_from_scores(scores: Mapping[str, float]) -> list[str]:
    """Metric names sorted best -> worst, with the same non-finite contract.

    Ties are broken by name (not by dict insertion order), so two rankers that both
    saturate produce orderings that are comparable rather than coincidentally equal.
    """
    names = list(scores)
    finite = sorted(
        (k for k in names if math.isfinite(scores[k])),
        key=lambda k: (-scores[k], k),
    )
    best = sorted(k for k in names if scores[k] == math.inf)
    worst = sorted(k for k in names if not math.isfinite(scores[k]) and scores[k] != math.inf)
    return best + finite + worst


__all__ = [
    "DegenerateRankingError",
    "assert_not_degenerate",
    "distinct_fraction",
    "order_from_scores",
    "ranks_from_scores",
]
