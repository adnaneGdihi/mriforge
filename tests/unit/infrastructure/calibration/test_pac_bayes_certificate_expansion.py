"""Unit tests for ``spectramr.infrastructure.calibration.pac_bayes_certificate``.

Covers:

* the McAllester complexity term — monotonicity in ``n``, vanishing at
  ``KL = 0`` is *not* the case (the log term keeps it > 0), input
  validation, and the documented range,
* the full bound — monotonicity, additivity of the TV-extension term,
  and input validation,
* the :class:`PACBayesCertificate` wrapper — non-vacuous flag, the
  ``PACBayesReport`` shape, and serialisation,
* edge cases — minimum ``n = 2`` is allowed, ``n = 1`` raises, ``δ``
  must lie strictly in ``(0, 1)``, KL must be non-negative.
"""

from __future__ import annotations

import math

import pytest

from spectramr.infrastructure.calibration.pac_bayes_certificate import (
    PACBayesCertificate,
    PACBayesReport,
    mcallester_complexity_term,
    pac_bayes_bound,
)


# ---------------------------------------------------------------------------
# mcallester_complexity_term
# ---------------------------------------------------------------------------


class TestMcAllesterComplexityTerm:
    """``slack = sqrt((KL + ln(2 sqrt n / delta)) / (2(n-1)))``."""

    def test_known_value_matches_closed_form(self) -> None:
        # n = 100, delta = 0.05, KL = 1.0
        kl, n, delta = 1.0, 100, 0.05
        expected = math.sqrt(
            (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * (n - 1))
        )
        assert mcallester_complexity_term(kl, n, delta) == pytest.approx(expected)

    def test_monotone_decreasing_in_n(self) -> None:
        """Slack tightens (shrinks) as the sample size grows."""
        values = [
            mcallester_complexity_term(1.0, n, 0.05)
            for n in (100, 1_000, 10_000, 100_000)
        ]
        for prev, curr in zip(values, values[1:]):
            assert curr < prev, f"slack must decrease in n; got {values}"

    def test_monotone_increasing_in_kl(self) -> None:
        """Higher posterior–prior divergence loosens the bound."""
        a = mcallester_complexity_term(0.0, 1000, 0.05)
        b = mcallester_complexity_term(1.0, 1000, 0.05)
        c = mcallester_complexity_term(10.0, 1000, 0.05)
        assert a < b < c

    def test_monotone_decreasing_in_delta(self) -> None:
        """Lower confidence ``δ`` (i.e. higher probability ``1-δ``) loosens the bound."""
        loose = mcallester_complexity_term(1.0, 1000, 0.5)
        medium = mcallester_complexity_term(1.0, 1000, 0.05)
        tight = mcallester_complexity_term(1.0, 1000, 0.001)
        assert loose < medium < tight

    def test_zero_kl_still_positive(self) -> None:
        """``KL = 0`` zeroes the divergence term but the log term keeps slack > 0."""
        slack = mcallester_complexity_term(0.0, 100, 0.05)
        assert slack > 0
        expected = math.sqrt(math.log(2.0 * math.sqrt(100) / 0.05) / (2.0 * 99))
        assert slack == pytest.approx(expected)

    def test_minimum_valid_n_is_two(self) -> None:
        """``n = 2`` is the smallest legal value (denominator ``2(n-1) = 2``)."""
        assert mcallester_complexity_term(0.0, 2, 0.05) > 0

    @pytest.mark.parametrize("bad_n", [-1, 0, 1])
    def test_n_below_two_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n must be"):
            mcallester_complexity_term(1.0, bad_n, 0.05)

    @pytest.mark.parametrize("bad_kl", [-1e-9, -1.0, -100.0])
    def test_negative_kl_raises(self, bad_kl: float) -> None:
        with pytest.raises(ValueError, match="kl_q_p must be"):
            mcallester_complexity_term(bad_kl, 100, 0.05)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.1, 2.0])
    def test_delta_outside_open_interval_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta must"):
            mcallester_complexity_term(1.0, 100, bad_delta)


# ---------------------------------------------------------------------------
# pac_bayes_bound
# ---------------------------------------------------------------------------


class TestPacBayesBound:
    """``bound = empirical_risk + slack + tv_tolerance * M``."""

    def test_equals_components_sum(self) -> None:
        kl, n, delta = 1.0, 1000, 0.05
        emp = 0.10
        tv = 0.02
        M = 1.0
        slack = mcallester_complexity_term(kl, n, delta)
        expected = emp + slack + tv * M
        assert pac_bayes_bound(emp, kl, n, delta, tv, M) == pytest.approx(expected)

    def test_monotone_decreasing_in_n(self) -> None:
        """Holding everything else fixed, more samples ⇒ tighter (smaller) bound."""
        bounds = [
            pac_bayes_bound(0.1, 1.0, n, 0.05) for n in (100, 1_000, 10_000, 100_000)
        ]
        for prev, curr in zip(bounds, bounds[1:]):
            assert curr < prev, f"bound must tighten as n grows; got {bounds}"

    def test_monotone_increasing_in_empirical_risk(self) -> None:
        a = pac_bayes_bound(0.05, 1.0, 1000, 0.05)
        b = pac_bayes_bound(0.10, 1.0, 1000, 0.05)
        c = pac_bayes_bound(0.20, 1.0, 1000, 0.05)
        assert a < b < c
        # Difference equals the risk gap (slack term cancels).
        assert b - a == pytest.approx(0.05)
        assert c - b == pytest.approx(0.10)

    def test_tv_term_adds_linearly(self) -> None:
        base = pac_bayes_bound(0.1, 1.0, 1000, 0.05, tv_tolerance=0.0, loss_bound_M=1.0)
        with_tv = pac_bayes_bound(
            0.1, 1.0, 1000, 0.05, tv_tolerance=0.1, loss_bound_M=1.0
        )
        assert with_tv - base == pytest.approx(0.1)

    def test_loss_bound_M_scales_tv_term(self) -> None:
        b1 = pac_bayes_bound(0.0, 0.0, 1000, 0.05, tv_tolerance=0.05, loss_bound_M=1.0)
        b2 = pac_bayes_bound(0.0, 0.0, 1000, 0.05, tv_tolerance=0.05, loss_bound_M=2.0)
        assert (b2 - b1) == pytest.approx(0.05 * (2.0 - 1.0))

    def test_zero_kl_zero_tv_reduces_to_log_slack(self) -> None:
        """With KL = 0 and τ = 0, the bound is just empirical + log slack."""
        emp, n, delta = 0.0, 1000, 0.05
        bound = pac_bayes_bound(emp, 0.0, n, delta, tv_tolerance=0.0)
        slack = math.sqrt(math.log(2.0 * math.sqrt(n) / delta) / (2.0 * (n - 1)))
        assert bound == pytest.approx(emp + slack)

    @pytest.mark.parametrize("bad_emp", [-1e-9, -0.1, -1.0])
    def test_negative_empirical_risk_raises(self, bad_emp: float) -> None:
        with pytest.raises(ValueError, match="empirical_risk"):
            pac_bayes_bound(bad_emp, 1.0, 100, 0.05)

    @pytest.mark.parametrize("bad_tv", [-1e-9, -0.5])
    def test_negative_tv_raises(self, bad_tv: float) -> None:
        with pytest.raises(ValueError, match="tv_tolerance"):
            pac_bayes_bound(0.1, 1.0, 100, 0.05, tv_tolerance=bad_tv)

    @pytest.mark.parametrize("bad_M", [0.0, -1.0, -1e-9])
    def test_non_positive_loss_bound_raises(self, bad_M: float) -> None:
        with pytest.raises(ValueError, match="loss_bound_M"):
            pac_bayes_bound(0.1, 1.0, 100, 0.05, loss_bound_M=bad_M)


# ---------------------------------------------------------------------------
# PACBayesCertificate (the wrapper)
# ---------------------------------------------------------------------------


class TestPacBayesCertificate:
    def test_certify_returns_report_with_expected_fields(self) -> None:
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=0.0)
        report = cert.certify(empirical_risk=0.05, kl_q_p=1.0, n=10_000)
        assert isinstance(report, PACBayesReport)
        assert report.empirical_risk == pytest.approx(0.05)
        assert report.kl_q_p == pytest.approx(1.0)
        assert report.n == 10_000
        assert report.delta == pytest.approx(0.05)
        assert report.tv_tolerance == pytest.approx(0.0)
        assert report.loss_bound_M == pytest.approx(1.0)
        # complexity_term + empirical_risk equals bound when tv = 0.
        assert report.bound == pytest.approx(report.empirical_risk + report.complexity_term)

    def test_non_vacuous_when_bound_below_M(self) -> None:
        # Very large n + tiny KL + tiny empirical risk ⇒ bound ≪ M.
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
        report = cert.certify(empirical_risk=0.0, kl_q_p=0.0, n=1_000_000)
        assert report.bound < 1.0
        assert report.is_non_vacuous is True

    def test_vacuous_when_bound_at_or_above_M(self) -> None:
        # Tiny n + huge KL ⇒ slack blows past M = 1.
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
        report = cert.certify(empirical_risk=0.5, kl_q_p=1e6, n=10)
        assert report.bound >= 1.0
        assert report.is_non_vacuous is False

    def test_tv_tolerance_is_threaded_through(self) -> None:
        base = PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=0.0)
        with_tv = PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=0.1)
        r_base = base.certify(0.1, 1.0, 1000)
        r_tv = with_tv.certify(0.1, 1.0, 1000)
        # The slack term is identical, only τ·M differs.
        assert r_tv.bound - r_base.bound == pytest.approx(0.1)

    def test_to_dict_round_trip(self) -> None:
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=2.0, tv_tolerance=0.01)
        report = cert.certify(empirical_risk=0.05, kl_q_p=1.0, n=10_000)
        d = report.to_dict()
        assert d["empirical_risk"] == pytest.approx(0.05)
        assert d["loss_bound_M"] == pytest.approx(2.0)
        assert d["tv_tolerance"] == pytest.approx(0.01)
        assert d["n"] == 10_000
        assert d["is_non_vacuous"] == report.is_non_vacuous

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.5, 2.0])
    def test_invalid_delta_at_construction_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            PACBayesCertificate(delta=bad_delta)

    @pytest.mark.parametrize("bad_M", [0.0, -1.0])
    def test_invalid_loss_bound_at_construction_raises(self, bad_M: float) -> None:
        with pytest.raises(ValueError, match="loss_bound_M"):
            PACBayesCertificate(delta=0.05, loss_bound_M=bad_M)

    def test_invalid_tv_at_construction_raises(self) -> None:
        with pytest.raises(ValueError, match="tv_tolerance"):
            PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=-0.01)

    def test_certify_propagates_n_validation(self) -> None:
        """Certificate must raise the same ``n ≥ 2`` error as the underlying bound."""
        cert = PACBayesCertificate(delta=0.05)
        with pytest.raises(ValueError, match="n must be"):
            cert.certify(empirical_risk=0.1, kl_q_p=1.0, n=1)

    def test_report_is_frozen_dataclass(self) -> None:
        cert = PACBayesCertificate(delta=0.05)
        report = cert.certify(0.1, 1.0, 1000)
        with pytest.raises((AttributeError, Exception)):
            report.bound = 0.0  # type: ignore[misc]

    def test_pac_bayes_bound_used_by_certify_matches_standalone(self) -> None:
        """The wrapper must use the same closed-form as ``pac_bayes_bound``."""
        cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=0.02)
        report = cert.certify(empirical_risk=0.1, kl_q_p=0.5, n=500)
        standalone = pac_bayes_bound(
            empirical_risk=0.1,
            kl_q_p=0.5,
            n=500,
            delta=0.05,
            tv_tolerance=0.02,
            loss_bound_M=1.0,
        )
        assert report.bound == pytest.approx(standalone)
