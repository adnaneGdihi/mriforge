"""Unit tests for ``mriforge.infrastructure.calibration.pathology_recall_certificate``.

Covers:

* the Hoeffding lower-bound (closed form, monotonicity, clamping to
  ``[0, 1]``, input validation),
* the ``required_empirical_recall`` inversion (sanity, infeasible
  target ⇒ ``+inf``),
* the :class:`PathologyRecallCertificate` audit gate (pass when
  bound clears target, fail otherwise) and the boundary behaviour at
  exactly the target (``>=`` ⇒ pass),
* the ``PathologyRecallReport`` data-class shape and serialisation.
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
# hoeffding_recall_lower_bound
# ---------------------------------------------------------------------------


class TestHoeffdingRecallLowerBound:
    """``bound = max(0, hat r - sqrt(ln(1/δ) / (2 n)))``."""

    def test_known_value_matches_closed_form(self) -> None:
        emp, n, delta = 0.99, 200, 0.05
        gap = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        expected = max(0.0, emp - gap)
        assert hoeffding_recall_lower_bound(emp, n, delta) == pytest.approx(expected)

    def test_documented_example_n200_delta005(self) -> None:
        """Docstring example: at δ=0.05, n=200, gap ≈ 0.087."""
        gap = math.sqrt(math.log(1.0 / 0.05) / (2.0 * 200))
        assert gap == pytest.approx(0.0865, abs=1e-3)
        # Empirical 0.987 ⇒ bound is ~0.90.
        assert hoeffding_recall_lower_bound(0.987, 200, 0.05) == pytest.approx(
            0.987 - gap
        )

    def test_monotone_increasing_in_n(self) -> None:
        """More positives ⇒ tighter (larger) lower bound for the same empirical recall."""
        values = [
            hoeffding_recall_lower_bound(0.95, n, 0.05)
            for n in (50, 100, 500, 5_000)
        ]
        for prev, curr in zip(values, values[1:]):
            assert curr > prev, f"bound must rise with n; got {values}"

    def test_monotone_increasing_in_empirical_recall(self) -> None:
        a = hoeffding_recall_lower_bound(0.80, 1000, 0.05)
        b = hoeffding_recall_lower_bound(0.90, 1000, 0.05)
        c = hoeffding_recall_lower_bound(0.99, 1000, 0.05)
        assert a < b < c

    def test_monotone_decreasing_in_delta(self) -> None:
        """Smaller δ ⇒ wider confidence interval ⇒ smaller lower bound."""
        loose = hoeffding_recall_lower_bound(0.95, 1000, 0.5)
        medium = hoeffding_recall_lower_bound(0.95, 1000, 0.05)
        tight = hoeffding_recall_lower_bound(0.95, 1000, 0.001)
        assert loose > medium > tight

    def test_bound_clamped_to_zero(self) -> None:
        """Even with tiny empirical recall and small n, the bound never goes negative."""
        # Empirical = 0.0, tiny n ⇒ raw bound is very negative; must clamp to 0.
        bound = hoeffding_recall_lower_bound(0.0, 5, 0.001)
        assert bound == 0.0

    def test_bound_never_exceeds_one(self) -> None:
        """Bound is at most ``empirical_recall ≤ 1`` (gap is non-negative)."""
        for n in (10, 100, 10_000):
            for emp in (0.5, 0.9, 1.0):
                bound = hoeffding_recall_lower_bound(emp, n, 0.05)
                assert 0.0 <= bound <= 1.0

    def test_perfect_recall_with_large_n_approaches_one(self) -> None:
        bound = hoeffding_recall_lower_bound(1.0, 1_000_000, 0.05)
        gap = math.sqrt(math.log(1.0 / 0.05) / (2.0 * 1_000_000))
        assert bound == pytest.approx(1.0 - gap)
        assert bound > 0.99

    @pytest.mark.parametrize("bad_emp", [-1e-9, -0.5, 1.0 + 1e-9, 1.5, 2.0])
    def test_empirical_recall_out_of_range_raises(self, bad_emp: float) -> None:
        with pytest.raises(ValueError, match="empirical_recall"):
            hoeffding_recall_lower_bound(bad_emp, 100, 0.05)

    @pytest.mark.parametrize("bad_n", [-10, -1, 0])
    def test_non_positive_n_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n_positives"):
            hoeffding_recall_lower_bound(0.9, bad_n, 0.05)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_delta_outside_open_interval_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            hoeffding_recall_lower_bound(0.9, 100, bad_delta)


# ---------------------------------------------------------------------------
# required_empirical_recall
# ---------------------------------------------------------------------------


class TestRequiredEmpiricalRecall:
    def test_inverts_lower_bound_at_minimum(self) -> None:
        """``hat r = target + gap`` should yield bound exactly equal to target."""
        target, n, delta = 0.90, 200, 0.05
        emp_needed = required_empirical_recall(target, n, delta)
        assert emp_needed < 1.0  # feasible
        bound = hoeffding_recall_lower_bound(emp_needed, n, delta)
        assert bound == pytest.approx(target, abs=1e-9)

    def test_monotone_decreasing_in_n(self) -> None:
        """The empirical bar drops as the sample size grows."""
        values = [
            required_empirical_recall(0.90, n, 0.05)
            for n in (100, 1_000, 10_000, 100_000)
        ]
        for prev, curr in zip(values, values[1:]):
            assert curr < prev, f"needed empirical must drop with n; got {values}"

    def test_infeasible_target_returns_inf(self) -> None:
        """If ``target + gap > 1`` then no empirical recall (≤1) can certify."""
        # Tiny n + tight δ + high target ⇒ infeasible.
        emp_needed = required_empirical_recall(0.99, 10, 1e-6)
        assert emp_needed == float("inf")

    def test_zero_target_is_always_feasible(self) -> None:
        """Target = 0 is certified by any empirical recall (gap clamped)."""
        emp_needed = required_empirical_recall(0.0, 100, 0.05)
        # We just require ``emp >= gap``; finite and ≤ 1.
        assert math.isfinite(emp_needed)
        assert 0.0 <= emp_needed <= 1.0

    @pytest.mark.parametrize("bad_target", [-1e-9, -0.5, 1.0 + 1e-9, 1.5])
    def test_invalid_target_raises(self, bad_target: float) -> None:
        with pytest.raises(ValueError, match="target_recall"):
            required_empirical_recall(bad_target, 100, 0.05)

    @pytest.mark.parametrize("bad_n", [-1, 0])
    def test_non_positive_n_raises(self, bad_n: int) -> None:
        with pytest.raises(ValueError, match="n_positives"):
            required_empirical_recall(0.9, bad_n, 0.05)

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_delta_raises(self, bad_delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            required_empirical_recall(0.9, 100, bad_delta)


# ---------------------------------------------------------------------------
# PathologyRecallCertificate
# ---------------------------------------------------------------------------


class TestPathologyRecallCertificate:
    def test_construction_validates_args(self) -> None:
        with pytest.raises(ValueError, match="pathology_class"):
            PathologyRecallCertificate(pathology_class="", target_recall=0.9)
        with pytest.raises(ValueError, match="target_recall"):
            PathologyRecallCertificate(
                pathology_class="infarct", target_recall=1.5
            )
        with pytest.raises(ValueError, match="target_recall"):
            PathologyRecallCertificate(
                pathology_class="infarct", target_recall=-0.1
            )
        with pytest.raises(ValueError, match="delta"):
            PathologyRecallCertificate(
                pathology_class="infarct", target_recall=0.9, delta=0.0
            )
        with pytest.raises(ValueError, match="delta"):
            PathologyRecallCertificate(
                pathology_class="infarct", target_recall=0.9, delta=1.0
            )

    def test_passes_when_bound_clears_target(self) -> None:
        """High empirical recall + many positives ⇒ certificate passes."""
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.90, delta=0.05
        )
        # 100k positives crushes the gap; empirical 0.95 easily certifies.
        report = cert.certify(empirical_recall=0.95, n_positives=100_000)
        assert report.is_certified is True
        assert report.certified_recall_lower_bound >= report.target_recall

    def test_fails_when_bound_misses_target(self) -> None:
        """Empirical recall below target ⇒ bound below target ⇒ fail."""
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.90, delta=0.05
        )
        # Empirical 0.80 cannot certify 0.90, even with huge n (bound ≈ 0.80).
        report = cert.certify(empirical_recall=0.80, n_positives=100_000)
        assert report.is_certified is False
        assert report.certified_recall_lower_bound < report.target_recall

    def test_fails_when_n_is_small_despite_perfect_empirical(self) -> None:
        """Perfect recall + tiny n still fails because the gap is too wide."""
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.95, delta=0.001
        )
        # n = 5, δ = 0.001 ⇒ gap ~ 0.83 ⇒ bound on emp=1.0 ~ 0.17.
        report = cert.certify(empirical_recall=1.0, n_positives=5)
        assert report.is_certified is False

    def test_borderline_exact_target_is_certified_inclusive(self) -> None:
        """At ``bound == target`` the gate uses ``>=`` ⇒ pass (inclusive)."""
        target, n, delta = 0.90, 1000, 0.05
        # Pick empirical = target + gap so the bound lands exactly on target.
        gap = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        emp = target + gap
        assert emp <= 1.0  # sanity for this n, δ
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=target, delta=delta
        )
        report = cert.certify(empirical_recall=emp, n_positives=n)
        # Bound clamps; assert near-equality and gate result.
        assert report.certified_recall_lower_bound == pytest.approx(target)
        assert report.is_certified is True  # ``>=`` is inclusive

    def test_report_fields_round_trip(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="meningioma", target_recall=0.85, delta=0.05
        )
        report = cert.certify(empirical_recall=0.99, n_positives=500)
        assert isinstance(report, PathologyRecallReport)
        assert report.pathology_class == "meningioma"
        assert report.empirical_recall == pytest.approx(0.99)
        assert report.n_positives == 500
        assert report.delta == pytest.approx(0.05)
        assert report.target_recall == pytest.approx(0.85)
        # to_dict serialises all fields.
        d = report.to_dict()
        assert d["pathology_class"] == "meningioma"
        assert d["n_positives"] == 500
        assert d["is_certified"] == report.is_certified

    def test_report_is_frozen_dataclass(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.9, delta=0.05
        )
        report = cert.certify(empirical_recall=0.99, n_positives=500)
        with pytest.raises((AttributeError, Exception)):
            report.is_certified = False  # type: ignore[misc]

    def test_certify_propagates_n_validation(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.9, delta=0.05
        )
        with pytest.raises(ValueError, match="n_positives"):
            cert.certify(empirical_recall=0.95, n_positives=0)

    def test_certify_propagates_empirical_validation(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.9, delta=0.05
        )
        with pytest.raises(ValueError, match="empirical_recall"):
            cert.certify(empirical_recall=1.5, n_positives=100)

    def test_required_empirical_recall_method_matches_free_function(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.90, delta=0.05
        )
        for n in (100, 1_000, 10_000):
            assert cert.required_empirical_recall(n) == pytest.approx(
                required_empirical_recall(0.90, n, 0.05)
            )

    def test_required_empirical_recall_method_infeasible(self) -> None:
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.99, delta=1e-6
        )
        assert cert.required_empirical_recall(10) == float("inf")

    def test_is_certified_flag_matches_bound_vs_target(self) -> None:
        """The gate is exactly ``bound >= target``; no hidden tolerance."""
        cert = PathologyRecallCertificate(
            pathology_class="infarct", target_recall=0.50, delta=0.05
        )
        # Sweep empirical recall over a grid and check consistency.
        for emp in (0.10, 0.30, 0.50, 0.70, 0.90):
            report = cert.certify(empirical_recall=emp, n_positives=10_000)
            expected = report.certified_recall_lower_bound >= report.target_recall
            assert report.is_certified is expected
