r"""Tests for the cross-site conformal aggregator.

Targets ``mriforge.infrastructure.services.conformal_aggregator_service`` —
idea 8 §8.2 of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.

Categories:

- Construction: empty sites / unknown rule / non-positive n_cal /
  negative quantile rejected
- ``max`` rule selects the largest per-site quantile
- ``weighted`` rule produces an inverse-of-N-weighted average; the
  closed-form result is verified
- Single-site call returns the input quantile (idempotent)
- Weights of report sum to 1
- Plan invariant: ``max`` rule's weights are uniform
"""

from __future__ import annotations

import pytest

from mriforge.infrastructure.services.conformal_aggregator_service import (
    SiteCalibration,
    aggregate_site_quantiles,
)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_empty_sites_rejected() -> None:
    """``sites=[]`` raises."""
    with pytest.raises(ValueError, match="empty"):
        aggregate_site_quantiles([])


def test_unknown_rule_rejected() -> None:
    """Only ``max`` / ``weighted`` accepted."""
    with pytest.raises(ValueError, match="rule"):
        aggregate_site_quantiles(
            [SiteCalibration("s1", quantile=1.0, n_calibration=10)],
            rule="foo",
        )


def test_zero_n_calibration_rejected() -> None:
    """Per-site ``n_calibration <= 0`` rejected."""
    with pytest.raises(ValueError, match="n_calibration"):
        aggregate_site_quantiles(
            [SiteCalibration("s1", quantile=1.0, n_calibration=0)]
        )


def test_negative_quantile_rejected() -> None:
    """Negative quantile rejected (band radii are non-negative)."""
    with pytest.raises(ValueError, match="quantile"):
        aggregate_site_quantiles(
            [SiteCalibration("s1", quantile=-0.1, n_calibration=10)]
        )


# ---------------------------------------------------------------------------
# Max rule
# ---------------------------------------------------------------------------


def test_max_rule_selects_largest() -> None:
    """``max`` rule returns the largest per-site quantile."""
    sites = [
        SiteCalibration("s1", quantile=0.4, n_calibration=10),
        SiteCalibration("s2", quantile=0.7, n_calibration=20),
        SiteCalibration("s3", quantile=0.2, n_calibration=5),
    ]
    report = aggregate_site_quantiles(sites, rule="max")
    assert report.global_quantile == 0.7
    assert report.rule == "max"
    assert report.contributing_sites == 3


def test_max_rule_weights_are_uniform() -> None:
    """``max`` rule reports uniform per-site weights (diagnostic only)."""
    sites = [
        SiteCalibration("s1", quantile=0.4, n_calibration=10),
        SiteCalibration("s2", quantile=0.7, n_calibration=99),
    ]
    report = aggregate_site_quantiles(sites, rule="max")
    assert report.weights == (0.5, 0.5)


# ---------------------------------------------------------------------------
# Weighted rule
# ---------------------------------------------------------------------------


def test_weighted_rule_sample_size_average() -> None:
    """Weighted rule = sample-size-weighted mean."""
    # n1=10, n2=30 → weights 0.25, 0.75
    sites = [
        SiteCalibration("s1", quantile=0.4, n_calibration=10),
        SiteCalibration("s2", quantile=0.8, n_calibration=30),
    ]
    report = aggregate_site_quantiles(sites, rule="weighted")
    expected = 0.25 * 0.4 + 0.75 * 0.8
    assert pytest.approx(report.global_quantile) == expected


def test_weighted_rule_weights_sum_to_one() -> None:
    """Weighted rule's reported weights sum to 1."""
    sites = [
        SiteCalibration("s1", quantile=0.4, n_calibration=10),
        SiteCalibration("s2", quantile=0.8, n_calibration=30),
        SiteCalibration("s3", quantile=0.6, n_calibration=60),
    ]
    report = aggregate_site_quantiles(sites, rule="weighted")
    assert pytest.approx(sum(report.weights)) == 1.0


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_single_site_returns_its_quantile() -> None:
    """Single-site federation collapses to the per-site quantile."""
    site = SiteCalibration("solo", quantile=0.65, n_calibration=100)
    for rule in ("max", "weighted"):
        report = aggregate_site_quantiles([site], rule=rule)
        assert pytest.approx(report.global_quantile) == 0.65
        assert report.contributing_sites == 1


# ---------------------------------------------------------------------------
# Conservativeness ordering
# ---------------------------------------------------------------------------


def test_max_rule_at_least_as_large_as_weighted() -> None:
    """``max`` rule is always ≥ ``weighted`` rule for the same input."""
    sites = [
        SiteCalibration("s1", quantile=0.4, n_calibration=10),
        SiteCalibration("s2", quantile=0.8, n_calibration=30),
        SiteCalibration("s3", quantile=0.6, n_calibration=60),
    ]
    q_max = aggregate_site_quantiles(sites, rule="max").global_quantile
    q_weighted = aggregate_site_quantiles(sites, rule="weighted").global_quantile
    assert q_max >= q_weighted
