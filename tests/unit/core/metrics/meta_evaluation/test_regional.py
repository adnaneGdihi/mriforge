"""Region-conditional evaluation: the pool is honest, and the gate actually fires.

The point of these tests is NOT "does it run". The whole stack ran before, and scored
the full FOV while the region packages sat unreachable behind their own green suites
(pitfall #16). So what is asserted here is that the region path **changes the answer**:
the metric object is not invoked where it is undefined, and the pool that reaches the
leaderboard is smaller than the pool that went in.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.meta_evaluation.regional import (  # noqa: E402
    build_regional_bundles,
    region_ids_in,
)
from spectramr.core.metrics.meta_evaluation.types import (  # noqa: E402
    DegradationSample,
    MetricSet,
)
from spectramr.core.metrics.regions.types import (  # noqa: E402
    RegionMask,
    RegionSet,
    RegionSource,
)

SIDE = 320


def _sample(content_id: str, theta: float, family: str = "gaussian_noise"):
    clean = torch.rand(1, SIDE, SIDE)
    return DegradationSample(
        clean=clean,
        degraded=(clean + theta * torch.randn(1, SIDE, SIDE) * 0.1).clamp(0, 1),
        degradation_family=family,
        severity={"theta": theta},
        content_id=content_id,
        seed=0,
    )


def _box(y0: int, x0: int, side: int, region_id: str, source: str) -> RegionMask:
    mask = torch.zeros(SIDE, SIDE, dtype=torch.bool)
    mask[y0 : y0 + side, x0 : x0 + side] = True
    return RegionMask(mask=mask, region_id=region_id, source=source, provenance={"src": "test"})


def _region_set(lesion_side: int = 40) -> RegionSet:
    return RegionSet(
        regions=(
            RegionSet.full_region(SIDE, SIDE),
            _box(50, 50, lesion_side, "path:lesion_any", RegionSource.PATHOLOGY),
            _box(50, 230, lesion_side, "control:lesion_any", RegionSource.CONTROL),
        )
    )


def _metric_set(keys: list[str]) -> MetricSet:
    from spectramr.core.metrics.registry import get_metric

    higher = {"psnr": True, "ssim": True, "ms_ssim": True, "mse": False, "mae": False}
    return MetricSet(
        metrics={k: get_metric(k) for k in keys},
        higher_is_better={k: higher.get(k, False) for k in keys},
    )


class TestTheGateActuallyFires:
    def test_a_metric_undefined_on_a_lesion_roi_is_absent_from_that_pool(self) -> None:
        """ms_ssim needs 161 px/side. A 40x40 fastMRI+ box is every lesion cell.

        ms_ssim is stubbed rather than resolved from the registry: the gate keys off the
        metric NAME, so a stub exercises the real policy, and the test then does not fail
        for the unrelated reason that torchmetrics cannot import in a given environment.
        """
        samples = [_sample("s0", t / 4) for t in range(5)]
        regions = {"s0": _region_set(lesion_side=40)}

        base = _metric_set(["psnr", "mse"])
        metric_set = MetricSet(
            metrics={**base.metrics, "ms_ssim": lambda p, t: 0.9},
            higher_is_better={**base.higher_is_better, "ms_ssim": True},
        )

        bundles = {b.region_id: b for b in build_regional_bundles(samples, metric_set, regions)}

        assert "ms_ssim" in bundles["full"].metric_set.metrics
        lesion = bundles["path:lesion_any"]
        assert "ms_ssim" not in lesion.metric_set.metrics
        assert "161" in lesion.excluded["ms_ssim"]
        # The pool that reaches the leaderboard is SMALLER, and says so.
        assert lesion.pool_size < bundles["full"].pool_size

    def test_the_metric_object_is_never_invoked_where_it_is_ineligible(self) -> None:
        """Not called-and-discarded: never called. Every metric returns *something* on a
        40px crop, so 'we called it and threw the number away' is one refactor away from
        'we called it and kept the number'."""
        calls: list[str] = []

        def spy(pred, target):
            calls.append("invoked")
            return 0.5

        metric_set = MetricSet(metrics={"ms_ssim": spy}, higher_is_better={"ms_ssim": True})
        samples = [_sample("s0", 0.5)]
        bundles = build_regional_bundles(
            samples, metric_set, {"s0": _region_set(40)}, region_ids=["path:lesion_any"]
        )

        assert bundles[0].pool_size == 0
        assert calls == [], "the gate declined ms_ssim but the metric ran anyway"

    def test_a_physics_metric_is_not_restrictable_on_any_region(self) -> None:
        samples = [_sample("s0", 0.5)]
        bundles = {
            b.region_id: b
            for b in build_regional_bundles(
                samples, _metric_set(["g_factor", "mse"]), {"s0": _region_set()}
            )
        }
        for region_id in ("full", "path:lesion_any"):
            assert "g_factor" not in bundles[region_id].metric_set.metrics
            assert "not_roi_restrictable" in bundles[region_id].excluded["g_factor"]


class TestThePoolIsAllOrNothing:
    def test_a_metric_eligible_on_only_some_instances_is_excluded_entirely(
        self,
    ) -> None:
        """The subtle one. Scoring a metric only where it fits hands the rankers a column
        correlated over the LARGE lesions only -- a silent selection effect, in exactly
        the cells the region axis exists to interrogate."""
        samples = [_sample("small", 0.5), _sample("big", 0.5)]
        regions = {
            "small": _region_set(lesion_side=40),  # below niqe's 96px floor
            "big": _region_set(lesion_side=200),  # above it
        }

        bundles = {
            b.region_id: b
            for b in build_regional_bundles(samples, _metric_set(["niqe", "mse"]), regions)
        }
        lesion = bundles["path:lesion_any"]

        assert "niqe" not in lesion.metric_set.metrics
        assert "1/2 instances" in lesion.excluded["niqe"], (
            "the exclusion must say how WIDELY it bit: cut on every instance is a policy "
            "decision, cut on 1 of 2 is a geometry effect, and they read differently."
        )
        assert "mse" in lesion.metric_set.metrics


class TestNoSilentFallbacks:
    def test_a_sample_with_no_region_set_raises(self) -> None:
        samples = [_sample("s0", 0.5), _sample("orphan", 0.5)]
        with pytest.raises(KeyError, match="no RegionSet"):
            build_regional_bundles(samples, _metric_set(["mse"]), {"s0": _region_set()})

    def test_a_region_absent_from_a_slice_shrinks_that_regions_dataset(self) -> None:
        """A lesion region exists on annotated slices only. The lesion bundle carries
        fewer samples than `full` -- which is why the cell power-gates on its own n."""
        plain = RegionSet(regions=(RegionSet.full_region(SIDE, SIDE),))
        samples = [_sample("annotated", 0.5), _sample("clean", 0.5)]
        regions = {"annotated": _region_set(), "clean": plain}

        bundles = {
            b.region_id: b for b in build_regional_bundles(samples, _metric_set(["mse"]), regions)
        }
        assert bundles["full"].n_samples == 2
        assert bundles["path:lesion_any"].n_samples == 1


class TestRegionIdsIn:
    def test_full_leads_and_the_rest_are_sorted(self) -> None:
        ids = region_ids_in({"s0": _region_set()})
        assert ids[0] == "full"
        assert ids == ("full", "control:lesion_any", "path:lesion_any")


class TestCoverage:
    def test_the_coverage_row_states_the_denominator(self) -> None:
        samples = [_sample("s0", 0.5)]
        bundles = build_regional_bundles(
            samples,
            _metric_set(["psnr", "mse", "ms_ssim", "g_factor"]),
            {"s0": _region_set(40)},
            region_ids=["path:lesion_any"],
        )
        row = bundles[0].coverage_row()
        assert row["region"] == "path:lesion_any"
        assert row["pool_size"] == 2  # psnr + mse survive; ms_ssim + g_factor do not
        assert row["n_excluded"] == 2
