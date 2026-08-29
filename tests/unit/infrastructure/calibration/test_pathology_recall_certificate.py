r"""Tests for the Pathology Recall Certificate (PRC) — ULF roadmap PR-23 / R4.

Targets ``mriforge.infrastructure.calibration.pathology_recall_certificate``.

Plan acceptance criterion (§"Tests"):

- *"Synthetic detector with controlled recall — verify certificate
  matches the analytic bound exactly within numerical tolerance."*
  Covered by ``test_bound_matches_closed_form`` and
  ``test_operational_consequence_n_200_delta_005``.

Categories:

- ``hoeffding_recall_lower_bound`` matches the closed form
- Bound is clamped to ``[0, 1]``
- Bound shrinks (Hoeffding gap grows) with smaller n / smaller δ
- ``required_empirical_recall`` inverts the bound; returns ``inf`` when
  the target is unattainable
- Certificate certifies / fails to certify on the documented operational example
- Invalid arguments rejected
- Report serialises to a JSON-friendly dict
"""

from __future__ import annotations

import math

import pytest

from mriforge.infrastructure.calibration.pathology_recall_certificate import (
    PathologyRecallCertificate,
    PathologyRecallReport,
    hoeffding_recall_lower_bound,
    required_empirical_recall,
)


# ---------------------------------------------------------------------------
# Closed-form Hoeffding bound
# ---------------------------------------------------------------------------


def test_bound_matches_closed_form() -> None:
    """``hat r - sqrt(ln(1/δ)/(2n))`` is the exact closed form."""
    n, delta = 200, 0.05
    hat_r = 0.95
    expected = hat_r - math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    assert pytest.approx(
        hoeffding_recall_lower_bound(hat_r, n, delta)
    ) == expected


def test_bound_clamped_to_zero_when_gap_exceeds_recall() -> None:
    """If ``gap > hat r`` the bound clamps to 0 (recall ≥ 0 is a free fact)."""
    out = hoeffding_recall_lower_bound(empirical_recall=0.1, n_positives=5, delta=0.05)
    assert out == 0.0


def test_bound_decreases_with_smaller_n() -> None:
    """Smaller n → larger Hoeffding gap → smaller certified bound."""
    high_n = hoeffding_recall_lower_bound(0.9, n_positives=10000, delta=0.05)
    low_n = hoeffding_recall_lower_bound(0.9, n_positives=10, delta=0.05)
    assert high_n > low_n


def test_bound_decreases_with_tighter_delta() -> None:
    """Smaller δ (more confident) → larger gap → smaller bound."""
    loose = hoeffding_recall_lower_bound(0.9, n_positives=200, delta=0.10)
    tight = hoeffding_recall_lower_bound(0.9, n_positives=200, delta=0.001)
    assert loose > tight


# ---------------------------------------------------------------------------
# required_empirical_recall (the inverse direction)
# ---------------------------------------------------------------------------


def test_required_empirical_recall_round_trip() -> None:
    """``hat r = required(target, n, δ)`` certifies exactly the target."""
    needed = required_empirical_recall(target_recall=0.9, n_positives=200, delta=0.05)
    assert needed < 1.0  # achievable
    bound = hoeffding_recall_lower_bound(needed, 200, 0.05)
    assert pytest.approx(bound, abs=1e-6) == 0.9


def test_required_empirical_recall_returns_inf_when_unattainable() -> None:
    """When target + gap > 1, no empirical recall can certify."""
    # Very small n, tight δ, high target → unattainable.
    out = required_empirical_recall(target_recall=0.99, n_positives=5, delta=1e-6)
    assert out == float("inf")


# ---------------------------------------------------------------------------
# Plan operational example: n=200, δ=0.05 → gap ≈ 0.087
# ---------------------------------------------------------------------------


def test_operational_consequence_n_200_delta_005() -> None:
    """Plan-quoted operational example: gap at n=200, δ=0.05 ≈ 0.087."""
    bound = hoeffding_recall_lower_bound(
        empirical_recall=1.0, n_positives=200, delta=0.05
    )
    expected_gap = math.sqrt(math.log(1.0 / 0.05) / (2.0 * 200))
    assert pytest.approx(bound) == 1.0 - expected_gap
    assert abs(expected_gap - 0.087) < 0.005


def test_required_recall_to_certify_90_at_n_200_is_987() -> None:
    """To certify ``r ≥ 0.90`` at ``n=200, δ=0.05`` need ``hat r ≈ 0.987``.

    This is the headline number the plan codifies for clinical claims.
    """
    needed = required_empirical_recall(target_recall=0.90, n_positives=200, delta=0.05)
    # Allow a small tolerance for the closed-form value.
    assert abs(needed - 0.987) < 0.005


# ---------------------------------------------------------------------------
# Certificate construction
# ---------------------------------------------------------------------------


def test_empty_pathology_class_rejected() -> None:
    """Pathology class must be non-empty."""
    with pytest.raises(ValueError, match="pathology_class"):
        PathologyRecallCertificate(pathology_class="", target_recall=0.9)


def test_target_recall_out_of_range_rejected() -> None:
    """``target_recall ∉ [0, 1]`` rejected."""
    with pytest.raises(ValueError, match="target_recall"):
        PathologyRecallCertificate("infarct", target_recall=1.5)


def test_delta_out_of_range_rejected() -> None:
    """``δ ∉ (0, 1)`` rejected."""
    with pytest.raises(ValueError, match="delta"):
        PathologyRecallCertificate("infarct", target_recall=0.9, delta=1.5)


def test_default_delta_is_documented() -> None:
    """Default δ = 0.05."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9)
    assert cert.delta == 0.05


# ---------------------------------------------------------------------------
# certify() flow
# ---------------------------------------------------------------------------


def test_certify_passes_when_empirical_clears_threshold() -> None:
    """Empirical ≥ ``target + gap`` → certificate is ``is_certified = True``."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9, delta=0.05)
    needed = cert.required_empirical_recall(n_positives=200)
    report = cert.certify(empirical_recall=needed, n_positives=200)
    assert report.is_certified is True


def test_certify_fails_when_empirical_below_threshold() -> None:
    """Empirical recall just below threshold → certificate fails."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9, delta=0.05)
    # Empirical recall = target itself → bound = target - gap < target → fail.
    report = cert.certify(empirical_recall=0.9, n_positives=200)
    assert report.is_certified is False


def test_report_records_inputs() -> None:
    """The report carries forward all inputs for the JSON dump."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9, delta=0.05)
    report = cert.certify(empirical_recall=0.95, n_positives=100)
    assert report.pathology_class == "infarct"
    assert report.empirical_recall == 0.95
    assert report.n_positives == 100
    assert report.delta == 0.05
    assert report.target_recall == 0.9


def test_report_to_dict_is_json_friendly() -> None:
    """``to_dict`` returns a flat dict of primitives."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9)
    report = cert.certify(0.95, 100)
    d = report.to_dict()
    for value in d.values():
        assert isinstance(value, (str, int, float, bool))


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------


def test_report_is_pathology_recall_report_instance() -> None:
    """Output type is :class:`PathologyRecallReport`."""
    cert = PathologyRecallCertificate("infarct", target_recall=0.9)
    report = cert.certify(0.95, 100)
    assert isinstance(report, PathologyRecallReport)
