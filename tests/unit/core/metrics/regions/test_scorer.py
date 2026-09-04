"""The scorer: T6 (the gate bites) and T9 (matched-pair parity).

T6 is the one that matters: an ineligible metric must **never be invoked**. Not
called-and-discarded -- never called. Every metric returns *something* when handed a
40x40 crop, so the only safe design is not to hand it one.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.outcome import MetricStatus  # noqa: E402
from spectramr.core.metrics.regions.scorer import (  # noqa: E402
    eligibility_table,
    score_region,
    score_region_set,
)
from spectramr.core.metrics.regions.types import (  # noqa: E402
    RegionMask,
    RegionSet,
    RegionSource,
)


def _full(h: int = 320, w: int = 320) -> RegionMask:
    return RegionSet.full_region(h, w)


def _lesion(side: int = 41, y: int = 10, x: int = 10) -> RegionMask:
    mask = torch.zeros(320, 320, dtype=torch.bool)
    mask[y : y + side, x : x + side] = True
    return RegionMask(
        mask=mask,
        region_id=f"path:lesion@{y},{x}",
        source=RegionSource.PATHOLOGY,
        provenance={"annotation": "fastmri_plus"},
    )


@pytest.fixture
def pair():
    g = torch.Generator().manual_seed(5)
    target = torch.rand(1, 1, 320, 320, generator=g)
    pred = (target + 0.05 * torch.randn(1, 1, 320, 320, generator=g)).clamp(0, 1)
    return pred, target


class TestT6TheGateBites:
    def test_an_ineligible_metric_is_never_invoked(self, pair) -> None:
        """The core anti-facade assertion.

        MS-SSIM on a 41x41 lesion ROI is undefined (it needs 161 px/side). The old
        engine upsampled and returned a confident score. The gate must not even hand
        the metric the data.
        """
        pred, target = pair
        calls: list[int] = []

        def never_call_me(*_a: object, **_k: object) -> float:
            calls.append(1)
            return 0.99  # a plausible-looking MS-SSIM

        outcome = score_region("ms_ssim", never_call_me, pred, target, _lesion(41))

        assert calls == [], "the gate invoked a metric it had declared ineligible"
        assert outcome.status is MetricStatus.NOT_APPLICABLE
        assert outcome.reason == "roi_min_side_below_metric_floor"
        assert math.isnan(outcome.value)

    def test_a_physics_metric_is_never_invoked_on_a_region(self, pair) -> None:
        pred, target = pair
        calls: list[int] = []

        def never_call_me(*_a: object, **_k: object) -> float:
            calls.append(1)
            return 1.4

        outcome = score_region("g_factor", never_call_me, pred, target, _lesion(200))
        assert calls == []
        assert outcome.status is MetricStatus.NOT_APPLICABLE
        assert outcome.reason == "physics_measurement_not_roi_restrictable"


class TestMapTierScoring:
    def test_a_map_tier_metric_scores_the_region_not_the_slice(self, pair) -> None:
        pred, target = pair
        whole = score_region("mse", lambda *_: 0.0, pred, target, _full())
        lesion = score_region("mse", lambda *_: 0.0, pred, target, _lesion(41))

        assert whole.status is MetricStatus.OK
        assert lesion.status is MetricStatus.OK
        # Different supports => different values (the restriction actually restricts).
        assert whole.value != pytest.approx(lesion.value, rel=1e-6)

    def test_the_map_tier_ignores_the_passed_metric_fn(self, pair) -> None:
        """Map-tier values come from the reducer, not from the metric callable.

        Passing a bogus callable must not change the answer -- otherwise the tier is
        not really map-then-mask.
        """
        pred, target = pair
        a = score_region("mse", lambda *_: 999.0, pred, target, _lesion(41))
        b = score_region("mse", lambda *_: -999.0, pred, target, _lesion(41))
        assert a.value == pytest.approx(b.value)


class TestCropTierScoring:
    def test_a_crop_tier_metric_receives_only_the_bbox(self, pair) -> None:
        pred, target = pair
        seen: list[tuple[int, ...]] = []

        def record_shape(p, t):
            seen.append(tuple(p.shape))
            return 0.5

        out = score_region("hfen", record_shape, pred, target, _lesion(64))
        assert out.status is MetricStatus.OK
        # Exactly the bbox -- no resample, no pad.
        assert seen == [(1, 1, 64, 64)]


class TestCrashIsNotIneligible:
    def test_a_crashing_metric_is_reported_as_crashed(self, pair) -> None:
        pred, target = pair

        def boom(*_a: object, **_k: object) -> float:
            raise RuntimeError("metric exploded")

        out = score_region("hfen", boom, pred, target, _lesion(64))
        assert out.status is MetricStatus.CRASHED
        assert out.reason == "RuntimeError"

    def test_a_nan_return_is_a_crash_not_a_score(self, pair) -> None:
        pred, target = pair
        out = score_region("hfen", lambda *_: float("nan"), pred, target, _lesion(64))
        assert out.status is MetricStatus.CRASHED
        assert out.reason == "NaNReturn"


class TestT9MatchedPairParity:
    """A lesion and its size-matched control MUST admit the same metric pool.

    Otherwise the paired comparison -- the whole point of the control -- silently
    compares different metric sets and any difference is an artifact of the pool.
    """

    def test_a_lesion_and_its_size_matched_control_agree_on_eligibility(self) -> None:
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS

        metrics = [s.registry_key for s in METRIC_SPECS]
        lesion = _lesion(41, y=10, x=10)
        control = RegionMask(  # same size, mirrored position
            mask=_lesion(41, y=10, x=269).mask,
            region_id="control:lesion@10,10",
            source=RegionSource.CONTROL,
            provenance={"paired_with": "path:lesion@10,10"},
        )

        elig_lesion = {
            e.metric
            for e in eligibility_table(metrics, RegionSet((_full(), lesion)))
            if e.eligible and e.region_id == lesion.region_id
        }
        elig_control = {
            e.metric
            for e in eligibility_table(metrics, RegionSet((_full(), control)))
            if e.eligible and e.region_id == control.region_id
        }
        assert elig_lesion == elig_control, (
            "a lesion and its size-matched control admit different metric pools; the "
            "paired comparison would be comparing different metrics."
        )


class TestEligibilityTableIsEmittableUpFront:
    def test_the_table_is_a_pure_function_of_policy_and_geometry(self) -> None:
        regions = RegionSet((_full(), _lesion(41)))
        rows = eligibility_table(["mse", "ms_ssim", "g_factor"], regions)
        assert len(rows) == 6  # 3 metrics x 2 regions
        by = {(r.metric, r.region_id): r for r in rows}
        assert by[("mse", "path:lesion@10,10")].eligible
        assert not by[("ms_ssim", "path:lesion@10,10")].eligible
        assert not by[("g_factor", "full")].eligible  # never, on any region


class TestScoreRegionSet:
    def test_every_region_gets_a_report(self, pair) -> None:
        pred, target = pair
        regions = RegionSet((_full(), _lesion(41)))
        reports = score_region_set(
            {"mse": lambda *_: 0.0, "ms_ssim": lambda *_: 0.9}, pred, target, regions
        )
        assert set(reports) == {"full", "path:lesion@10,10"}
        lesion = reports["path:lesion@10,10"]
        assert [o.metric for o in lesion.scored] == ["mse"]
        assert [o.metric for o in lesion.not_applicable] == ["ms_ssim"]
        assert not lesion.crashed
