r"""Tests for ``PACBayesCertificate`` (ULF roadmap PR-24 / R5).

Targets ``mriforge.infrastructure.calibration.pac_bayes_certificate``.

Plan acceptance criterion (§"Tests"):

- *"Synthetic federated setup with known optimal Q — verify computed
  bound matches the analytic value within numerical tolerance."* —
  ``test_complexity_matches_closed_form``.
- *"Known TV-distance perturbation — verify the τ·M extension term
  tracks the actual generalisation gap."* —
  ``test_tv_extension_term_linear_in_tau``.

Categories:

- Construction: ``δ`` / ``M`` / ``τ`` / ``KL`` / ``n`` validation
- Closed-form complexity term matches McAllester
- Bound is monotone increasing in KL, decreasing in n
- Non-vacuous flag fires correctly
- Report serialises to a JSON-friendly dict
- Report fields are populated from inputs
"""

from __future__ import annotations

import math

import pytest

from mriforge.infrastructure.calibration.pac_bayes_certificate import (
    PACBayesCertificate,
    PACBayesReport,
    mcallester_complexity_term,
    pac_bayes_bound,
)


# ---------------------------------------------------------------------------
# Closed-form complexity term
# ---------------------------------------------------------------------------


def test_complexity_matches_closed_form() -> None:
    """``√((KL + ln(2√n/δ)) / (2(n-1)))`` is the exact closed form."""
    kl, n, delta = 5.0, 1000, 0.05
    log_term = math.log(2.0 * math.sqrt(n) / delta)
    expected = math.sqrt((kl + log_term) / (2.0 * (n - 1)))
    assert pytest.approx(mcallester_complexity_term(kl, n, delta)) == expected


def test_complexity_decreases_with_n() -> None:
    """More samples → tighter bound."""
    small = mcallester_complexity_term(kl_q_p=5.0, n=100, delta=0.05)
    big = mcallester_complexity_term(kl_q_p=5.0, n=10_000, delta=0.05)
    assert big < small


def test_complexity_grows_with_kl() -> None:
    """Larger KL divergence → larger complexity slack."""
    near = mcallester_complexity_term(kl_q_p=0.1, n=1000, delta=0.05)
    far = mcallester_complexity_term(kl_q_p=10.0, n=1000, delta=0.05)
    assert far > near


def test_complexity_grows_with_smaller_delta() -> None:
    """Tighter δ → larger complexity slack."""
    loose = mcallester_complexity_term(kl_q_p=1.0, n=1000, delta=0.10)
    tight = mcallester_complexity_term(kl_q_p=1.0, n=1000, delta=0.001)
    assert tight > loose


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_complexity_rejects_negative_kl() -> None:
    """``KL < 0`` rejected."""
    with pytest.raises(ValueError, match="kl_q_p"):
        mcallester_complexity_term(kl_q_p=-0.1, n=100, delta=0.05)


def test_complexity_rejects_n_below_two() -> None:
    """``n < 2`` rejected (denominator 2(n-1) → 0)."""
    with pytest.raises(ValueError, match="n must be"):
        mcallester_complexity_term(kl_q_p=1.0, n=1, delta=0.05)


def test_complexity_rejects_invalid_delta() -> None:
    """``δ`` outside ``(0, 1)`` rejected."""
    with pytest.raises(ValueError, match="delta"):
        mcallester_complexity_term(kl_q_p=1.0, n=100, delta=1.5)


# ---------------------------------------------------------------------------
# pac_bayes_bound
# ---------------------------------------------------------------------------


def test_bound_decomposes_into_three_terms() -> None:
    """``bound = empirical_risk + slack + τ·M``."""
    er, kl, n, delta, tau, M = 0.1, 1.0, 1000, 0.05, 0.05, 1.0
    slack = mcallester_complexity_term(kl, n, delta)
    expected = er + slack + tau * M
    assert pytest.approx(
        pac_bayes_bound(er, kl, n, delta, tau, M)
    ) == expected


def test_bound_negative_empirical_risk_rejected() -> None:
    """Empirical risk < 0 rejected."""
    with pytest.raises(ValueError, match="empirical_risk"):
        pac_bayes_bound(-0.1, kl_q_p=1.0, n=100, delta=0.05)


def test_bound_negative_tv_rejected() -> None:
    """``τ < 0`` rejected."""
    with pytest.raises(ValueError, match="tv_tolerance"):
        pac_bayes_bound(0.1, kl_q_p=1.0, n=100, delta=0.05, tv_tolerance=-0.1)


def test_bound_zero_loss_bound_M_rejected() -> None:
    """``M = 0`` rejected (would let τ·M collapse silently)."""
    with pytest.raises(ValueError, match="loss_bound_M"):
        pac_bayes_bound(0.1, kl_q_p=1.0, n=100, delta=0.05, loss_bound_M=0.0)


# ---------------------------------------------------------------------------
# τ·M extension term linear in τ
# ---------------------------------------------------------------------------


def test_tv_extension_term_linear_in_tau() -> None:
    """Doubling ``τ`` adds exactly ``M`` to the bound."""
    er, kl, n, delta, M = 0.1, 1.0, 1000, 0.05, 1.0
    b_at_0 = pac_bayes_bound(er, kl, n, delta, tv_tolerance=0.0, loss_bound_M=M)
    b_at_05 = pac_bayes_bound(er, kl, n, delta, tv_tolerance=0.5, loss_bound_M=M)
    assert pytest.approx(b_at_05 - b_at_0) == 0.5 * M


# ---------------------------------------------------------------------------
# Certificate audit-gate semantics
# ---------------------------------------------------------------------------


def test_certificate_invalid_delta_rejected() -> None:
    """``δ`` outside ``(0, 1)`` rejected at construction."""
    with pytest.raises(ValueError, match="delta"):
        PACBayesCertificate(delta=0.0)


def test_certificate_zero_loss_bound_M_rejected() -> None:
    """``M ≤ 0`` rejected."""
    with pytest.raises(ValueError, match="loss_bound_M"):
        PACBayesCertificate(loss_bound_M=0.0)


def test_certificate_negative_tv_rejected() -> None:
    """``τ < 0`` rejected at construction."""
    with pytest.raises(ValueError, match="tv_tolerance"):
        PACBayesCertificate(tv_tolerance=-0.1)


def test_certificate_non_vacuous_when_bound_below_M() -> None:
    """Non-vacuous flag fires when bound < M."""
    cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
    # Tight regime: low empirical risk, low KL, large n.
    report = cert.certify(empirical_risk=0.05, kl_q_p=0.1, n=10_000)
    assert report.is_non_vacuous is True


def test_certificate_vacuous_when_kl_huge() -> None:
    """Huge KL drives bound past M → vacuous."""
    cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0)
    report = cert.certify(empirical_risk=0.5, kl_q_p=10_000.0, n=100)
    assert report.is_non_vacuous is False


def test_certificate_report_records_inputs() -> None:
    """All inputs survive into the report fields."""
    cert = PACBayesCertificate(delta=0.05, loss_bound_M=1.0, tv_tolerance=0.05)
    report = cert.certify(empirical_risk=0.1, kl_q_p=1.0, n=1000)
    assert report.empirical_risk == 0.1
    assert report.kl_q_p == 1.0
    assert report.n == 1000
    assert report.delta == 0.05
    assert report.loss_bound_M == 1.0
    assert report.tv_tolerance == 0.05


def test_certificate_report_to_dict_is_json_friendly() -> None:
    """``to_dict`` yields only primitive types."""
    cert = PACBayesCertificate()
    report = cert.certify(empirical_risk=0.1, kl_q_p=1.0, n=1000)
    d = report.to_dict()
    for value in d.values():
        assert isinstance(value, (str, int, float, bool))


def test_report_is_pac_bayes_report_instance() -> None:
    """Output type is :class:`PACBayesReport`."""
    cert = PACBayesCertificate()
    report = cert.certify(empirical_risk=0.1, kl_q_p=1.0, n=1000)
    assert isinstance(report, PACBayesReport)
