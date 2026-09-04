"""Tests for the L2⁺ betting / e-value anytime-valid certification.

These pin the executable counterparts of the Lean theorems in
``docs/lean/Sim2Rank/Sim2Rank/Betting.lean`` (wealth recursion, finite Markov /
e-value validity, variance-adaptive confidence sequence, paired-increment
certification).
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation.betting import (
    betting_confidence_sequence,
    betting_wealth,
    certify_betting_order,
    evalue_tail_mass,
    kendall_concordance_increments,
)


def test_wealth_empty_is_one() -> None:
    # Lean wealth_zero: empty product = 1.
    assert betting_wealth([], 0.0, 0.5) == 1.0


def test_wealth_recursion() -> None:
    # Lean wealth_succ: each factor multiplies in. (1+0.5)(1+0.5) = 2.25.
    assert betting_wealth([1.0, 1.0], 0.0, 0.5) == pytest.approx(2.25)


def test_wealth_nonneg_under_bounded_bets() -> None:
    # |lam*(x-eta)| <= 1 keeps every factor >= 0 (Lean wealth_nonneg).
    w = betting_wealth([1.0, -1.0, 0.5, -0.5], 0.0, 0.5)
    assert w >= 0.0


def test_evalue_tail_mass_at_most_alpha_for_evalue() -> None:
    # An e-value (mean <= 1) puts mass <= alpha above 1/alpha (Lean evalue_test_valid).
    alpha = 0.1  # threshold 1/alpha = 10
    # uniform weights summing to 1; one outcome at 12 (>=10), rest at 0.
    values = [12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    weights = [0.1] * 10  # mean = 1.2 ... not an e-value; make it one:
    values = [10.0] + [0.0] * 9
    weights = [0.1] * 10  # mean = 1.0 (e-value); mass above 10 is exactly 0.1 = alpha
    mass = evalue_tail_mass(values, weights, alpha)
    assert mass <= alpha + 1e-12


def test_cs_brackets_constant_data() -> None:
    # Constant data at 0.5: the CS must contain 0.5 and stay inside [-1, 1].
    lo, hi = betting_confidence_sequence([0.5] * 80, 0.05, lo=-1.0, hi=1.0)
    assert lo <= 0.5 <= hi
    assert -1.0 <= lo <= hi <= 1.0


def test_cs_shrinks_with_more_data() -> None:
    # More low-variance samples -> tighter (variance-adaptive) interval.
    short = betting_confidence_sequence([0.3] * 20, 0.05, lo=-1.0, hi=1.0)
    long = betting_confidence_sequence([0.3] * 400, 0.05, lo=-1.0, hi=1.0)
    assert (long[1] - long[0]) <= (short[1] - short[0])


def test_cs_excludes_far_values() -> None:
    # Strongly concentrated near +0.9 -> the CS must exclude 0 for large n.
    lo, _hi = betting_confidence_sequence([0.9] * 300, 0.05, lo=-1.0, hi=1.0)
    assert lo > 0.0


def test_cs_is_genuine_two_sided_one_minus_alpha() -> None:
    """The CS must be a true (1-alpha) TWO-sided interval, not a (1-2alpha) one.

    The interval intersects two one-sided e-processes; by the union bound a true
    1-alpha two-sided CS must threshold EACH side at ``2/alpha`` (alpha/2 per
    side), not ``1/alpha``. For constant data the variance-adaptive bets saturate
    at the cap ``1/(2*range)``, so the half-width is analytic:

        delta = (thr**(1/n) - 1) / cap,   thr = per-side threshold.

    This lets us pin the per-side threshold exactly (up to one grid step) and
    distinguish the correct ``2/alpha`` construction from the naive ``1/alpha``.
    """
    n, alpha, c = 80, 0.05, 0.5
    lo_b, hi_b = -1.0, 1.0
    cap = 1.0 / (2.0 * (hi_b - lo_b))
    lo, hi = betting_confidence_sequence([c] * n, alpha, lo=lo_b, hi=hi_b, grid=201)

    delta_two_sided = ((2.0 / alpha) ** (1.0 / n) - 1.0) / cap  # correct
    delta_naive = ((1.0 / alpha) ** (1.0 / n) - 1.0) / cap      # buggy (1-2a)
    grid_step = (hi_b - lo_b) / (201 - 1)
    tol = grid_step + 1e-9

    # Matches the two-sided construction...
    assert abs((c - lo) - delta_two_sided) <= tol
    assert abs((hi - c) - delta_two_sided) <= tol
    # ...and is clearly NOT the naive 1/alpha interval.
    assert abs((c - lo) - delta_naive) > tol


def test_kendall_increments_track_severity() -> None:
    # A metric increasing with severity yields all +1 concordance increments.
    inc = kendall_concordance_increments([0.1, 0.2, 0.3, 0.4], [0.0, 0.25, 0.5, 0.75])
    assert inc == [1.0, 1.0, 1.0]
    # A flat metric yields all 0.
    flat = kendall_concordance_increments([1.0, 1.0, 1.0], [0.0, 0.5, 1.0])
    assert flat == [0.0, 0.0]


def test_certify_orders_clear_pair() -> None:
    # Metric "good" tracks severity (+1 increments); "bad" is anti-correlated (-1).
    good = [1.0] * 60
    bad = [-1.0] * 60
    order = certify_betting_order({"good": good, "bad": bad}, alpha=0.05)
    assert order.is_certified("good", "bad")
    assert order.n_ambiguous == 0


def test_certify_withholds_indistinguishable_pair() -> None:
    # Two metrics with identical increment streams cannot be ordered.
    same = [1.0, -1.0, 1.0, -1.0] * 5
    order = certify_betting_order({"a": list(same), "b": list(same)}, alpha=0.05)
    assert order.n_certified == 0
    assert ("a", "b") in order.ambiguous


def test_certify_tolerates_nonfinite_increments() -> None:
    # crash→NaN contract — characterization lock. In the wired pipeline the
    # increments are signs in {-1, 0, 1} (always finite), but as a public API
    # certify_betting_order may receive raw NaN/inf. It must NOT crash and must
    # NOT spuriously certify: the variance-adaptive bet collapses on non-finite
    # data, so the paired CS stays wide and the pair is ambiguous (conservative).
    order = certify_betting_order(
        {"a": [float("nan"), 1.0, 1.0], "b": [-1.0, -1.0, float("inf")]}, alpha=0.05
    )
    assert order.n_certified == 0
    assert ("a", "b") in order.ambiguous


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_alpha_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        betting_confidence_sequence([0.5, 0.5], alpha)
    with pytest.raises(ValueError):
        evalue_tail_mass([1.0], [1.0], alpha)


def test_cs_empty_returns_full_range() -> None:
    assert betting_confidence_sequence([], 0.05, lo=-1.0, hi=1.0) == (-1.0, 1.0)


def test_cs_no_survivor_returns_full_range_not_point() -> None:
    # Data whose mean sits OUTSIDE [lo, hi] rejects every candidate mean -> no
    # survivor. The honest answer is the full (no-information) range, NOT a
    # zero-width / inverted ``(mean, mean)`` point that falsely claims certainty
    # (critique L1). Pre-fix this returned (5.0, 1.0) — an inverted interval.
    lo, hi = betting_confidence_sequence([5.0] * 40, 0.05, lo=-1.0, hi=1.0)
    assert (lo, hi) == (-1.0, 1.0)


def test_evalue_tail_mass_rejects_negative_value() -> None:
    # The <= alpha guarantee is conditional on an e-value (nonnegative statistic);
    # a negative entry means the input is not an e-value, so fail loud (critique M5).
    with pytest.raises(ValueError, match="nonnegative"):
        evalue_tail_mass([-0.1, 1.0], [0.5, 0.5], alpha=0.1)


def test_evalue_tail_mass_rejects_non_probability_weights() -> None:
    # The mass / <= alpha bound needs ``weights`` to be a probability law
    # (nonnegative, sum 1); otherwise the returned number is meaningless (crit. L2).
    with pytest.raises(ValueError, match="nonnegative"):
        evalue_tail_mass([1.0, 1.0], [1.5, -0.5], alpha=0.1)
    with pytest.raises(ValueError, match="sum to 1"):
        evalue_tail_mass([1.0, 1.0], [0.3, 0.3], alpha=0.1)


# ──────────────────────────────────────────────────────────────────────
# Vectorised ``_max_wealth``: the scalar recursion is the oracle.
# ──────────────────────────────────────────────────────────────────────


def _max_wealth_scalar(
    x: list[float], m: float, bets: list[float], sign: float
) -> float:
    """The pre-vectorisation scalar recursion, kept as the reference oracle.

    ``_max_wealth`` now runs the same product through ``np.cumprod``. Because
    ``cumprod`` accumulates left-to-right in this exact order, the two must
    agree bit-for-bit, not merely approximately — so these assertions use
    ``==`` deliberately.
    """
    wealth = 1.0
    peak = 1.0
    for xt, lam in zip(x, bets, strict=True):
        wealth *= 1.0 + sign * lam * (xt - m)
        if wealth > peak:
            peak = wealth
    return peak


@pytest.mark.parametrize("n", [0, 1, 2, 17, 200])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_max_wealth_is_bit_identical_to_scalar_recursion(n: int, sign: float) -> None:
    import numpy as np

    from spectramr.core.metrics.meta_evaluation.betting import _max_wealth

    rng = np.random.default_rng(4242 + n)
    x = list(rng.standard_normal(n))
    bets = list(np.abs(rng.standard_normal(n)) * 0.1)
    for m in (-1.0, 0.0, 0.3333333333, 1.0):
        assert _max_wealth(x, m, bets, sign) == _max_wealth_scalar(x, m, bets, sign)


def test_max_wealth_rejects_length_mismatch() -> None:
    """The ``zip(..., strict=True)`` contract survives the vectorisation."""
    from spectramr.core.metrics.meta_evaluation.betting import _max_wealth

    with pytest.raises(ValueError, match="equal length"):
        _max_wealth([1.0, 2.0], 0.0, [0.1], 1.0)


def test_max_wealth_never_reports_below_one() -> None:
    """Ville's running maximum starts at the initial wealth of 1."""
    from spectramr.core.metrics.meta_evaluation.betting import _max_wealth

    # Every factor < 1, so the raw cumulative product decays monotonically.
    assert _max_wealth([-1.0] * 8, 0.0, [0.4] * 8, 1.0) == 1.0


def test_confidence_sequence_accepts_arrays_and_sequences() -> None:
    """The sweep hoists conversion to arrays; plain lists must still work."""
    import numpy as np

    x = list(np.random.default_rng(9).standard_normal(64) * 0.3)
    assert betting_confidence_sequence(x, 0.05, lo=-2.0, hi=2.0) == (
        betting_confidence_sequence(np.asarray(x), 0.05, lo=-2.0, hi=2.0)
    )
