"""Tests for the Pathology Recall Certificate (ULF-PR-23 / R4).

Locks the Hoeffding lower-bound math, the Tier-1 audit hook, and the
JSON certificate emit path.

Per the spec doc: at ``delta=0.05, n=200``, certifying ``recall ≥ 0.90``
requires empirical recall ≥ ~0.987. These tests pin those numbers
exactly so a future contributor can't silently weaken the bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mriforge.infrastructure.certification.pathology_recall import (
    PathologyRecallCertificate,
    build_pathology_recall_certificate,
    hoeffding_recall_lower_bound,
    minimum_n_for_target_recall,
    pathology_test_set_size_sufficient,
    write_certificates_json,
)


# ---------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------


def test_hoeffding_bound_perfect_recall_loses_about_0_087_at_n_200_delta_005() -> None:
    """At ``n=200, δ=0.05`` the Hoeffding gap is ~0.087.

    Spec doc operational note: this is THE bar. The test pins it within
    1e-3 numerical tolerance.
    """
    bound = hoeffding_recall_lower_bound(empirical_recall=1.0, n=200, delta=0.05)
    expected_gap = math.sqrt(math.log(20) / 400.0)
    assert bound == pytest.approx(1.0 - expected_gap, abs=1e-9)
    # Sanity: gap is in the ~0.087 ballpark.
    assert 0.085 < (1.0 - bound) < 0.090


def test_hoeffding_bound_is_zero_clipped() -> None:
    """Empirical recall well below the gap produces a clipped 0 (not negative)."""
    bound = hoeffding_recall_lower_bound(empirical_recall=0.01, n=10, delta=0.05)
    assert bound == 0.0


def test_hoeffding_bound_validates_inputs() -> None:
    with pytest.raises(ValueError, match="empirical_recall"):
        hoeffding_recall_lower_bound(1.5, 100, 0.05)
    with pytest.raises(ValueError, match="n must be positive"):
        hoeffding_recall_lower_bound(0.9, 0, 0.05)
    with pytest.raises(ValueError, match="delta must be in"):
        hoeffding_recall_lower_bound(0.9, 100, 1.5)


def test_minimum_n_solves_for_target() -> None:
    """``minimum_n_for_target_recall`` is the inverse of the Hoeffding bound."""
    # At δ=0.05, target=0.90, empirical=0.987 → n should be ~ 200.
    n_required = minimum_n_for_target_recall(
        target_recall=0.90, empirical_recall=0.987, delta=0.05,
    )
    # The spec's "≥ 0.987 at n=200" comes from the same algebra — must round
    # to ~200 (a touch under or over due to integer ceil + numerical slack).
    assert 195 <= n_required <= 205


def test_minimum_n_is_infinity_when_empirical_too_low() -> None:
    """If empirical recall ≤ target, no n can certify."""
    out = minimum_n_for_target_recall(
        target_recall=0.90, empirical_recall=0.85, delta=0.05,
    )
    assert out == math.inf


# ---------------------------------------------------------------------
# Certificate dataclass
# ---------------------------------------------------------------------


def test_certificate_certified_when_bound_meets_target() -> None:
    cert = build_pathology_recall_certificate(
        "lesion", empirical_recall=0.99, n=500, delta=0.05, target_recall=0.90,
    )
    assert isinstance(cert, PathologyRecallCertificate)
    assert cert.certified is True
    assert cert.certified_lower_bound >= cert.target_recall


def test_certificate_not_certified_when_n_too_small() -> None:
    cert = build_pathology_recall_certificate(
        "lesion", empirical_recall=0.95, n=50, delta=0.05, target_recall=0.90,
    )
    assert cert.certified is False
    assert cert.certified_lower_bound < cert.target_recall


def test_certificate_round_trips_through_dict() -> None:
    cert = build_pathology_recall_certificate(
        "tumor", empirical_recall=0.99, n=1000, delta=0.05, target_recall=0.90,
    )
    d = cert.to_dict()
    assert d["pathology_class"] == "tumor"
    assert d["certified"] is True
    assert d["certified_lower_bound"] >= 0.90


# ---------------------------------------------------------------------
# Tier-1 audit hook
# ---------------------------------------------------------------------


def test_audit_passes_when_n_sufficient() -> None:
    audit = pathology_test_set_size_sufficient(
        "lesion", empirical_recall=0.99, n=1000,
        target_recall=0.90, delta=0.05,
    )
    assert audit.passed is True
    assert "PASS" in audit.message


def test_audit_fails_when_n_insufficient() -> None:
    audit = pathology_test_set_size_sufficient(
        "lesion", empirical_recall=0.95, n=50,
        target_recall=0.90, delta=0.05,
    )
    assert audit.passed is False
    assert "FAIL" in audit.message


def test_audit_flags_impossible_certification() -> None:
    """If empirical recall ≤ target, the audit says certification is impossible."""
    audit = pathology_test_set_size_sufficient(
        "lesion", empirical_recall=0.85, n=10_000,
        target_recall=0.90, delta=0.05,
    )
    assert audit.passed is False
    assert "impossible" in audit.message


# ---------------------------------------------------------------------
# JSON emit
# ---------------------------------------------------------------------


def test_write_certificates_json_round_trips(tmp_path: Path) -> None:
    certs = [
        build_pathology_recall_certificate(
            "lesion", empirical_recall=0.99, n=500,
        ),
        build_pathology_recall_certificate(
            "tumor", empirical_recall=0.95, n=50,
        ),
    ]
    out = write_certificates_json(certs, tmp_path / "pathology.json")
    payload = json.loads(out.read_text())
    assert len(payload) == 2
    classes = {entry["pathology_class"] for entry in payload}
    assert classes == {"lesion", "tumor"}
