"""Tests for the L5 conformal-risk-control layer.

These pin the executable counterparts of the Lean theorems in
``docs/lean/Sim2Rank/Sim2Rank/Conformal.lean`` (calibration rule, finite-sample
risk bound, up-set well-posedness, per-metric selection badge).
"""

from __future__ import annotations

import pytest

from spectramr.core.metrics.meta_evaluation.conformal import (
    conformal_metric_selection,
    crc_admissible,
    crc_risk_bound,
    select_conformal_threshold,
)


def test_admissible_rule_matches_closed_form() -> None:
    # (sum losses + B)/(n+1) <= alpha  (Lean crc_full_sample_mean_le).
    losses = [0.1] * 20
    n = len(losses)
    val = (sum(losses) + 1.0) / (n + 1)
    assert crc_admissible(losses, alpha=val) is True
    assert crc_admissible(losses, alpha=val - 1e-6) is False


def test_admissible_with_small_loss_bound() -> None:
    # B=0: pure calibration mean over n+1 (the strict / zero-slack case).
    losses = [0.05] * 10
    assert crc_admissible(losses, alpha=0.05, loss_bound=0.0) is True


def test_risk_bound_closed_form_and_limit() -> None:
    # (n+1)*alpha/n  (Lean conformal_risk_control); -> alpha as n -> inf.
    assert crc_risk_bound(100, 0.1) == pytest.approx(101 * 0.1 / 100)
    big = crc_risk_bound(10_000_000, 0.1)
    assert abs(big - 0.1) < 1e-6
    assert crc_risk_bound(50, 0.1) > 0.1  # always slightly above the target


def test_select_threshold_picks_least_conservative_admissible() -> None:
    # Loss is nonincreasing in lambda; the admissible set is an up-set
    # (Lean crc_threshold_upset) so the FIRST admissible threshold is returned.
    thresholds = [0.0, 1.0, 2.0, 3.0]
    # at lambda=0 loss is high (0.9), dropping to 0.0 by lambda=3.
    loss_at = [
        [0.9] * 10,  # mean 0.9 -> not admissible at alpha 0.2
        [0.5] * 10,  # mean 0.5 -> not admissible
        [0.1] * 10,  # (1.0+1)/11 = 0.18 <= 0.2 -> admissible (first one)
        [0.0] * 10,
    ]
    res = select_conformal_threshold(thresholds, loss_at, alpha=0.2, loss_bound=1.0)
    assert res.admissible
    assert res.threshold == 2.0
    assert res.controlled


def test_select_threshold_none_admissible_returns_most_conservative() -> None:
    thresholds = [0.0, 1.0]
    loss_at = [[0.9] * 5, [0.8] * 5]  # never clears a tiny alpha
    res = select_conformal_threshold(thresholds, loss_at, alpha=0.01, loss_bound=1.0)
    assert not res.admissible
    assert res.threshold == 1.0  # most conservative


def test_metric_selection_badges() -> None:
    # A low-loss metric is admissible; a high-loss one is not.
    losses = {"good": [0.02] * 30, "bad": [0.9] * 30}
    badges = conformal_metric_selection(losses, alpha=0.1, loss_bound=1.0)
    assert badges["good"].controlled
    assert not badges["bad"].controlled
    # Both carry the same finite-sample risk bound (same n).
    assert badges["good"].risk_bound == pytest.approx(badges["bad"].risk_bound)


def test_risk_bound_rejects_bad_n() -> None:
    with pytest.raises(ValueError):
        crc_risk_bound(0, 0.1)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.5])
def test_invalid_alpha_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        crc_admissible([0.1], alpha)
    with pytest.raises(ValueError):
        crc_risk_bound(10, alpha)


# --- crash→NaN contract: calibration losses can be NaN (metric crashed on a case) ---


def test_crc_admissible_clamps_nonfinite_to_max_loss() -> None:
    # A crashed (NaN) calibration case is conservatively counted at the worst-case
    # loss B, NOT dropped (critique A4): crashes are not missing-at-random (they hit
    # the hardest cases), so dropping them shrinks n and makes the rule *easier*,
    # voiding the distribution-free guarantee. clamped=[0.1, 1.0, 0.2] ->
    # (1.3 + 1)/4 = 0.575 > 0.5 -> NOT admissible — even though the same losses with
    # the crash *dropped* (finite=[0.1, 0.2]) would have passed at ≈0.433 <= 0.5.
    assert crc_admissible([0.1, float("nan"), 0.2], alpha=0.5, loss_bound=1.0) is False
    # At a looser level the conservatively-penalised metric can still be admissible.
    assert crc_admissible([0.1, float("nan"), 0.2], alpha=0.7, loss_bound=1.0) is True


def test_crc_admissible_all_nonfinite_not_admissible() -> None:
    # All crashes -> clamped to all-B -> (n*B + B)/(n+1) = B > alpha -> not
    # admissible (no crash). Conservative: a fully-broken metric is never certified.
    assert crc_admissible([float("nan"), float("inf")], alpha=0.9) is False


def test_metric_selection_clamps_nan_to_max_loss() -> None:
    # n_calib counts the FULL set (crash clamped to B=1.0), not a shrunken finite
    # subset (critique A4). clamped=[0.1, 1.0, 0.2] -> mean = 1.3/3 ≈ 0.4333.
    badges = conformal_metric_selection(
        {"m": [0.1, float("nan"), 0.2]}, alpha=0.5, loss_bound=1.0
    )
    b = badges["m"]
    assert b.calib_mean == pytest.approx(1.3 / 3)  # crash counted at max loss
    assert b.n_calib == 3  # full calibration set, crash included
    # (1.3 + 1)/4 = 0.575 > 0.5 -> conservatively NOT controlled.
    assert not b.controlled


def test_metric_selection_all_nan_is_max_loss_not_crash() -> None:
    # All-crash -> clamped to all-B -> not controlled, finite (max-loss) mean and a
    # finite risk bound. The "neutral, no crash" intent holds; the metric is just
    # conservatively un-certified rather than silently dropped to an empty set.
    badges = conformal_metric_selection({"m": [float("nan")] * 3}, alpha=0.5)
    b = badges["m"]
    assert b.n_calib == 3
    assert not b.controlled
    assert b.calib_mean == pytest.approx(1.0)  # all max loss
    assert b.risk_bound == pytest.approx(crc_risk_bound(3, 0.5))  # finite, defined
