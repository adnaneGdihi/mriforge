"""Tests for the Dice-risk RCPS certificate (B-1.7)."""

from __future__ import annotations

import math

import pytest

from mriforge.infrastructure.calibration.dice_risk_certificate import (
    DiceRiskCertificate,
    hoeffding_risk_upper_bound,
    rcps_threshold,
)


def test_ucb_above_empirical_and_clamped() -> None:
    ucb = hoeffding_risk_upper_bound(0.1, n=100, delta=0.05)
    assert ucb > 0.1  # gap added
    gap = math.sqrt(math.log(1 / 0.05) / (2 * 100))
    assert abs(ucb - (0.1 + gap)) < 1e-9
    # clamps to loss_bound
    assert hoeffding_risk_upper_bound(0.99, n=2, delta=0.01) <= 1.0


def test_ucb_gap_shrinks_with_n() -> None:
    assert hoeffding_risk_upper_bound(0.1, 50, 0.05) > hoeffding_risk_upper_bound(
        0.1, 5000, 0.05
    )


def test_ucb_validates() -> None:
    with pytest.raises(ValueError):
        hoeffding_risk_upper_bound(1.5, 10, 0.05)
    with pytest.raises(ValueError):
        hoeffding_risk_upper_bound(0.1, 0, 0.05)
    with pytest.raises(ValueError):
        hoeffding_risk_upper_bound(0.1, 10, 1.0)


def test_rcps_threshold_selects_smallest_controlling_lambda() -> None:
    # Monotone non-increasing risk vs lambda; large n so the Hoeffding gap is small.
    curve = [(0.0, 0.40), (0.25, 0.20), (0.5, 0.08), (0.75, 0.02), (1.0, 0.0)]
    lam = rcps_threshold(curve, alpha=0.1, delta=0.05, n=10000)
    # risk 0.08 at lambda 0.5 with a tiny gap is < 0.1 -> 0.5 controls; 0.20 does not
    assert lam == 0.5


def test_rcps_threshold_inf_when_uncontrollable() -> None:
    curve = [(0.0, 0.9), (0.5, 0.8), (1.0, 0.7)]
    assert rcps_threshold(curve, alpha=0.1, delta=0.05, n=100) == float("inf")


def test_certificate_certifies_low_risk_large_n() -> None:
    cert = DiceRiskCertificate(alpha=0.2, delta=0.05)
    rep = cert.certify(empirical_risk=0.05, n=5000)
    assert rep.is_certified
    assert rep.certified_risk_upper_bound <= 0.2
    assert rep.certified_risk_upper_bound > 0.05


def test_certificate_rejects_high_risk() -> None:
    cert = DiceRiskCertificate(alpha=0.1, delta=0.05)
    rep = cert.certify(empirical_risk=0.5, n=200)
    assert not rep.is_certified
    assert rep.to_dict()["empirical_risk"] == 0.5
