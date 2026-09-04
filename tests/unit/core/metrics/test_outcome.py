"""Tests for the metric outcome vocabulary.

The whole point of this module is that a NaN can no longer mean three different
things. These tests pin that: an OK outcome cannot carry NaN, a non-OK outcome
cannot carry a number, and a crash stays distinguishable from a not-applicable.
"""

from __future__ import annotations

import math

import pytest

from spectramr.core.metrics.outcome import (
    MetricNotApplicableError,
    MetricOutcome,
    MetricOutcomeReport,
    MetricStatus,
    NotApplicableReason,
)


class TestMetricOutcome:
    def test_ok_carries_the_value(self) -> None:
        o = MetricOutcome.ok("ssim", 0.87)
        assert o.status is MetricStatus.OK
        assert o.value == pytest.approx(0.87)
        assert o.reason is None

    def test_ok_outcome_cannot_carry_nan(self) -> None:
        """The core invariant: a NaN is never a measurement."""
        with pytest.raises(ValueError, match="cannot carry NaN"):
            MetricOutcome(metric="ssim", value=math.nan, status=MetricStatus.OK)

    def test_non_ok_outcome_cannot_carry_a_number(self) -> None:
        """A not-applicable metric has nothing to report -- it must not smuggle a value."""
        with pytest.raises(ValueError, match="must carry NaN"):
            MetricOutcome(
                metric="ms_ssim",
                value=0.5,
                status=MetricStatus.NOT_APPLICABLE,
                reason="roi_too_small",
            )

    def test_non_ok_outcome_requires_a_reason(self) -> None:
        with pytest.raises(ValueError, match="needs a reason"):
            MetricOutcome(metric="ms_ssim", value=math.nan, status=MetricStatus.NOT_APPLICABLE)

    def test_not_applicable_records_a_machine_readable_reason(self) -> None:
        o = MetricOutcome.not_applicable(
            "ms_ssim", NotApplicableReason.ROI_TOO_SMALL_SIDE, "41 < 161"
        )
        assert o.status is MetricStatus.NOT_APPLICABLE
        assert math.isnan(o.value)
        assert o.reason == str(NotApplicableReason.ROI_TOO_SMALL_SIDE)
        assert "41" in (o.detail or "")

    def test_crashed_preserves_the_exception_type(self) -> None:
        o = MetricOutcome.crashed("niqe", RuntimeError("boom"))
        assert o.status is MetricStatus.CRASHED
        assert o.reason == "RuntimeError"
        assert "boom" in (o.detail or "")

    def test_a_nan_return_is_classified_as_a_crash_not_a_result(self) -> None:
        """A metric that silently returns NaN has a bug; it has not measured anything."""
        o = MetricOutcome.from_value("lpips", float("nan"))
        assert o.status is MetricStatus.CRASHED
        assert o.reason == "NaNReturn"

    def test_a_real_return_is_ok(self) -> None:
        assert MetricOutcome.from_value("lpips", 0.2).status is MetricStatus.OK


class TestNotApplicableIsNotACrash:
    def test_the_two_are_distinguishable(self) -> None:
        """The bug B4 fixes: `except Exception -> NaN` made these identical."""
        na = MetricOutcome.not_applicable(
            "ms_ssim", NotApplicableReason.IMAGE_TOO_SMALL_FOR_METRIC, "40x40 < 161"
        )
        crash = MetricOutcome.crashed("ms_ssim", RuntimeError("kernel size"))
        # Both are NaN -- which is exactly why the old code could not tell them apart.
        assert math.isnan(na.value) and math.isnan(crash.value)
        # But the status channel keeps them separable.
        assert na.status is not crash.status

    def test_error_carries_reason_and_detail(self) -> None:
        exc = MetricNotApplicableError(
            "banding_region_mse",
            NotApplicableReason.MISSING_REQUIRED_ROI_MASK,
            "no mask supplied",
        )
        assert exc.reason is NotApplicableReason.MISSING_REQUIRED_ROI_MASK
        assert exc.metric == "banding_region_mse"
        assert "no mask supplied" in str(exc)


class TestMetricOutcomeReport:
    @staticmethod
    def _report() -> MetricOutcomeReport:
        return MetricOutcomeReport(
            outcomes=(
                MetricOutcome.ok("ssim", 0.9),
                MetricOutcome.not_applicable(
                    "ms_ssim", NotApplicableReason.ROI_TOO_SMALL_SIDE, "41 < 161"
                ),
                MetricOutcome.crashed("niqe", RuntimeError("boom")),
            )
        )

    def test_partitions_the_three_statuses(self) -> None:
        r = self._report()
        assert [o.metric for o in r.scored] == ["ssim"]
        assert [o.metric for o in r.not_applicable] == ["ms_ssim"]
        assert [o.metric for o in r.crashed] == ["niqe"]

    def test_values_view_is_backwards_compatible(self) -> None:
        """Legacy callers still get `metric -> float`, NaN for anything unscored."""
        vals = self._report().values
        assert vals["ssim"] == pytest.approx(0.9)
        assert math.isnan(vals["ms_ssim"])
        assert math.isnan(vals["niqe"])

    def test_raise_if_crashed_names_the_crashed_metric(self) -> None:
        with pytest.raises(RuntimeError, match="niqe"):
            self._report().raise_if_crashed()

    def test_raise_if_crashed_is_silent_when_only_not_applicable(self) -> None:
        """A not-applicable metric is an expected outcome and must not fail the run."""
        MetricOutcomeReport(
            outcomes=(
                MetricOutcome.ok("ssim", 0.9),
                MetricOutcome.not_applicable(
                    "ms_ssim", NotApplicableReason.ROI_TOO_SMALL_SIDE, "41 < 161"
                ),
            )
        ).raise_if_crashed()

    def test_to_rows_exposes_the_reason_channel(self) -> None:
        rows = {r["metric"]: r for r in self._report().to_rows()}
        assert rows["ms_ssim"]["status"] == "not_applicable"
        assert rows["niqe"]["status"] == "crashed"
        assert rows["ssim"]["reason"] == ""


def test_data_range_unresolved_is_a_declared_reason():
    """Range-sensitive metrics need a declared reason, not free text.

    Added with the #180 fix: PSNR/SSIM/ClinicalSSIM refuse to score when the
    target honours no normalization contract. Routing that through
    `NotApplicableReason` rather than a message string is what keeps the
    exclusion table groupable — 'why is val_psnr NaN on this cohort' has to be
    answerable without reading a traceback.
    """
    from spectramr.core.metrics.outcome import (
        MetricNotApplicableError,
        NotApplicableReason,
    )

    reason = NotApplicableReason.DATA_RANGE_UNRESOLVED
    assert reason.value == "data_range_unresolved_for_range_sensitive_metric"

    err = MetricNotApplicableError("psnr", reason, "target spans [0, 2479]")
    assert err.metric == "psnr"
    assert err.reason is reason
    assert "not applicable" in str(err)
