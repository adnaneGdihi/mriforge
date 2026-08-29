"""Betting / e-value anytime-valid certification (Layer L2⁺).

The powered certification in :mod:`powered` inverts the *Hoeffding* bound: it is
non-adaptive (ignores the per-content rank-agreement variance) and fixed-``n``
(the severity sweep cannot stop early without invalidating the guarantee). This
module is its variance-adaptive, anytime-valid counterpart — the executable
shadow of the Lean module ``Sim2Rank.Betting``
(``docs/lean/Sim2Rank/Sim2Rank/Betting.lean``):

* :func:`betting_wealth` is the wealth process ``K_t = ∏ (1 + λ_s (x_s − η))``
  of Lean ``wealth`` / ``wealth_succ`` (same product recursion, base case 1).
* :func:`evalue_tail_mass` is the finite Markov / e-value bound of Lean
  ``finite_markov`` / ``evalue_test_valid``: a nonnegative statistic with mean
  ``≤ 1`` puts mass ``≤ α`` above ``1/α``.
* :func:`betting_confidence_sequence` builds a time-uniform confidence sequence
  for a bounded mean by the Waudby-Smith–Ramdas hedged-capital construction; the
  anytime-validity is exactly Lean ``betting_cs_anytime_covers`` (Ville's
  inequality on the wealth supermartingale).
* :func:`certify_betting_order` is the betting analogue of
  :func:`powered.certify_partial_order`: it certifies a pairwise ordering when the
  paired-increment confidence sequence excludes zero.

NumPy-free, torch-free: pure ``math`` so it imports anywhere, matching
:mod:`powered`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BettingPartialOrder",
    "betting_confidence_sequence",
    "betting_wealth",
    "certify_betting_order",
    "evalue_tail_mass",
    "kendall_concordance_increments",
]


def betting_wealth(increments: Sequence[float], eta: float, lam: float) -> float:
    """Terminal wealth ``K_T = ∏_s (1 + lam · (x_s − eta))`` (Lean ``wealth``).

    The empty product is ``1`` (Lean ``wealth_zero``); each factor multiplies the
    running capital (Lean ``wealth_succ``). With ``|lam·(x−eta)| ≤ 1`` every
    factor is nonnegative, so the process stays nonnegative (Lean
    ``wealth_nonneg``).
    """
    wealth = 1.0
    for x in increments:
        wealth *= 1.0 + lam * (x - eta)
    return wealth


def evalue_tail_mass(values: Sequence[float], weights: Sequence[float], alpha: float) -> float:
    """Mass that a nonnegative statistic ``K`` (``values``) puts at or above ``1/α``.

    This is the empirical side of Lean ``evalue_test_valid``: for an e-value
    (``K ≥ 0``, weighted mean ``≤ 1``) the returned mass is provably ``≤ α``.
    Computed by the finite Markov inequality of Lean ``finite_markov`` — pure
    weighted-sum algebra.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(values) != len(weights):
        raise ValueError("values and weights must be aligned")
    # The <= alpha guarantee is conditional on ``values`` being an e-value: a
    # nonnegative statistic. A negative entry means the caller did not pass an
    # e-value, so the returned mass would be a meaningless number dressed as a
    # validity certificate — fail loud instead (critique M5).
    if any(v < 0.0 for v in values):
        raise ValueError(
            "evalue_tail_mass expects a nonnegative statistic (an e-value); got a negative value"
        )
    # The mass interpretation (and the <= alpha bound) requires ``weights`` to be a
    # probability law: nonnegative and summing to 1. Validate it rather than assume
    # it (critique L2) — an unnormalised/signed weight silently voids the guarantee.
    if any(w < 0.0 for w in weights):
        raise ValueError("weights must be nonnegative (a probability law)")
    wsum = sum(weights)
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1 (a probability law); got {wsum}")
    thr = 1.0 / alpha
    return sum(w for v, w in zip(values, weights, strict=True) if v >= thr)


def _running_variance_bets(x: Sequence[float], lo: float, hi: float, alpha: float) -> list[float]:
    """Predictable, variance-adaptive bets ``λ_t`` (Waudby-Smith–Ramdas).

    Each ``λ_t`` is computed only from ``x_0 … x_{t-1}`` (predictable) and capped at
    ``1 / (2·(hi − lo))`` so every wealth factor ``1 + λ_t·(x_t − m)`` lies in
    ``[1/2, 3/2] > 0`` for any candidate mean ``m ∈ [lo, hi]`` — the boundedness
    Lean ``wealth_nonneg`` assumes.
    """
    rng = max(hi - lo, 1e-12)
    cap = 1.0 / (2.0 * rng)
    bets: list[float] = []
    run_sum = 0.0
    run_sq = 0.0
    ln_term = 2.0 * math.log(2.0 / alpha)
    for t in range(len(x)):
        if t == 0:
            var = rng * rng / 4.0  # uninformative prior = range²/4 (Popoviciu)
        else:
            mean = run_sum / t
            var = run_sq / t - mean * mean
            var = max(var, 1e-6)
        denom = var * (t + 1) * math.log(t + 2)
        lam = math.sqrt(ln_term / denom) if denom > 0 else cap
        bets.append(min(cap, lam))
        run_sum += x[t]
        run_sq += x[t] * x[t]
    return bets


def _max_wealth(x: Sequence[float], m: float, bets: Sequence[float], sign: float) -> float:
    """Running supremum of the one-sided wealth (Ville's running maximum).

    ``np.cumprod`` accumulates left-to-right in exactly the order the scalar
    loop multiplied, so every partial product -- and hence the supremum -- is
    bit-identical to the sequential form it replaces.

    This is the hot path of :func:`betting_confidence_sequence`: the grid sweep
    calls it twice per candidate mean (2x ``grid``, i.e. 402 calls per
    sequence at the default), and it measured 97.7% of that call. Callers in
    the sweep pass arrays so the ``asarray`` below is a no-op; passing plain
    sequences still works.
    """
    xa = np.asarray(x, dtype=np.float64)
    ba = np.asarray(bets, dtype=np.float64)
    if xa.shape != ba.shape:
        # Mirrors the ``zip(..., strict=True)`` contract this replaced.
        raise ValueError(f"x and bets must have equal length; got {xa.shape} and {ba.shape}.")
    if xa.size == 0:
        return 1.0
    return max(1.0, float(np.cumprod(1.0 + sign * ba * (xa - m)).max()))


def betting_confidence_sequence(
    x: Sequence[float],
    alpha: float = 0.05,
    *,
    lo: float = -1.0,
    hi: float = 1.0,
    grid: int = 201,
) -> tuple[float, float]:
    """Anytime-valid confidence sequence for the mean of bounded data ``x ∈ [lo, hi]``.

    Returns ``(L, U)``: a genuine **two-sided** ``1 − α`` confidence sequence —
    the true mean lies in ``[L, U]`` at all times with probability ``≥ 1 − α``.

    The interval intersects two one-sided e-processes (one testing ``mean > m``,
    one testing ``mean < m``). Each is a level-``α/2`` Ville test — its
    running-maximum wealth is thresholded at ``2/α`` — so by the union bound the
    two-sided miscoverage is ``≤ α/2 + α/2 = α``. (Thresholding each side at
    ``1/α`` instead would only give a ``1 − 2α`` interval — the union bound then
    sums to ``2α`` — which is what an earlier version did.) Time-uniform coverage
    of each side is Lean ``betting_cs_anytime_covers`` (Ville's inequality).
    Variance-adaptive: on low-variance data the interval is far tighter than the
    Hoeffding interval ``mean ± (hi−lo)·sqrt(ln(2/α)/(2n))`` that :mod:`powered`
    inverts.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(x)
    if n == 0:
        return (lo, hi)
    bets = _running_variance_bets(x, lo, hi, alpha)
    # Per-side threshold 2/α (α/2 per one-sided test): the union of the two
    # one-sided miscoverages is then ≤ α, so [L, U] is a true 1−α two-sided CS.
    thr = 2.0 / alpha
    survivors: list[float] = []
    # Convert once rather than per grid point: the sweep makes 2x ``grid``
    # calls, and ``np.asarray`` on an ndarray is a no-op inside ``_max_wealth``.
    x_arr = np.asarray(x, dtype=np.float64)
    bets_arr = np.asarray(bets, dtype=np.float64)
    for k in range(grid):
        m = lo + (hi - lo) * k / (grid - 1)
        up = _max_wealth(x_arr, m, bets_arr, +1.0)  # tests "mean > m"
        down = _max_wealth(x_arr, m, bets_arr, -1.0)  # tests "mean < m"
        if up < thr and down < thr:
            survivors.append(m)
    if not survivors:
        # No candidate mean survives (e.g. non-finite data collapses the bets, or
        # the grid is too coarse). The honest anytime-valid answer is "no
        # information" — the full bounded range — NOT a zero-width ``(mean, mean)``
        # point interval, which would falsely claim certainty and could let
        # ``certify_betting_order`` certify a pair off a collapsed CS (critique L1).
        return (lo, hi)
    return (survivors[0], survivors[-1])


@dataclass
class BettingPartialOrder:
    """A partial order certified by paired-increment confidence sequences."""

    alpha: float
    certified: list[tuple[str, str]]
    ambiguous: list[tuple[str, str]]
    confidence_sequences: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_certified(self) -> int:
        return len(self.certified)

    @property
    def n_ambiguous(self) -> int:
        return len(self.ambiguous)

    def is_certified(self, better: str, worse: str) -> bool:
        return (better, worse) in self.certified


def certify_betting_order(
    per_metric_increments: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
    *,
    lo: float = -1.0,
    hi: float = 1.0,
) -> BettingPartialOrder:
    """Certify pairwise orderings via paired betting confidence sequences.

    ``per_metric_increments[m]`` is the sequence of bounded concordance increments
    of metric ``m`` with severity (each in ``[lo, hi]``; see
    :func:`kendall_concordance_increments`). For each pair ``(a, b)`` the paired
    difference ``d_i = x^a_i − x^b_i ∈ [lo−hi, hi−lo]`` has a confidence sequence;
    the pair is certified in the direction whose CS excludes ``0``, otherwise it is
    ambiguous. Anytime-valid, so the sweep may stop the instant a pair is certified.

    Level: each pair uses the two-sided ``1 − α`` CS of
    :func:`betting_confidence_sequence`, so under a true tie (``μ_d = 0``) the
    probability of certifying *either* (wrong) direction is ``≤ α`` (``≤ α/2`` per
    direction). This is a **per-pair** guarantee — no family-wise correction is
    applied across the ``O(M²)`` pairs, so the expected number of spurious edges
    under a global null grows with the pair count; apply a multiplicity correction
    (or combine the e-values) downstream if a family-wise guarantee is required.
    """
    names = list(per_metric_increments)
    cs: dict[str, tuple[float, float]] = {
        m: betting_confidence_sequence(per_metric_increments[m], alpha, lo=lo, hi=hi) for m in names
    }
    drange = hi - lo
    certified: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xa, xb = per_metric_increments[a], per_metric_increments[b]
            n = min(len(xa), len(xb))
            d = [xa[k] - xb[k] for k in range(n)]
            lo_d, hi_d = betting_confidence_sequence(d, alpha, lo=-drange, hi=drange)
            if lo_d > 0.0:
                certified.append((a, b))
            elif hi_d < 0.0:
                certified.append((b, a))
            else:
                ambiguous.append((a, b))
    return BettingPartialOrder(
        alpha=alpha, certified=certified, ambiguous=ambiguous, confidence_sequences=cs
    )


def kendall_concordance_increments(
    values: Sequence[float], severities: Sequence[float]
) -> list[float]:
    """Per-adjacent-pair concordance signs of a metric trajectory with severity.

    Returns ``sign((value_{t+1} − value_t)·(severity_{t+1} − severity_t)) ∈ {−1, 0, 1}``
    for each consecutive pair, the bounded ``[−1, 1]`` concordance increment the
    betting confidence sequence consumes (the kernel of Lean's Kendall U-statistic).
    A metric that tracks severity yields increments concentrated at ``+1``.
    """
    if len(values) != len(severities):
        raise ValueError("values and severities must be aligned")
    out: list[float] = []
    for t in range(len(values) - 1):
        dv = values[t + 1] - values[t]
        ds = severities[t + 1] - severities[t]
        prod = dv * ds
        out.append(1.0 if prod > 0 else (-1.0 if prod < 0 else 0.0))
    return out
