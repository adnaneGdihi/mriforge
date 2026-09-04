"""Tests for the Otsu tissue-segmentation agreement metrics.

These metrics segment the prediction AND the target with the same deterministic
segmenter and compare the two label maps (the cohort has no segmentation labels).
The contract they must honour:

1. A perfect prediction scores perfectly (Dice 1.0, HD95 0.0).
2. A degraded prediction scores strictly worse — otherwise the metric is a facade.
3. They NEVER return NaN. ``pipelines/train.py`` accumulates validation metrics with
   a bare running sum and no non-finite guard, so one NaN poisons the whole eval.
4. Each is registered and declares an optimization direction.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.tissue_segmentation import (
    MIN_FOREGROUND_FRACTION,
    N_TISSUE_CLASSES,
    OtsuTissueSegmenter,
)

_SEG_METRICS = (
    "tissue_dice",
    "brain_mask_dice",
    "tissue_volume_similarity",
    "tissue_hd95",
)


def _brain_phantom(size: int = 64, shift: int = 0) -> torch.Tensor:
    """A [1, 1, H, W] phantom in [-1, 1] with three concentric tissue bands.

    Mirrors the data the arms actually see: minmax-normalised to [-1, 1] (NOT [0, 1]),
    mostly background, with distinct intensity plateaus so a 3-class Otsu partition is
    well-posed. ``shift`` grows the structure, changing the tissue volumes.
    """
    img = torch.full((size, size), -1.0)
    yy, xx = torch.meshgrid(
        torch.arange(size).float(), torch.arange(size).float(), indexing="ij"
    )
    r = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2).sqrt()
    img[r < 26 + shift] = -0.2  # outer band  (CSF-like)
    img[r < 18 + shift] = 0.3  # middle band (GM-like)
    img[r < 9 + shift] = 0.9  # core        (WM-like)
    return img[None, None]


class TestOtsuTissueSegmenter:
    def test_segments_into_background_plus_three_tissue_classes(self):
        seg = OtsuTissueSegmenter().segment(_brain_phantom())
        assert seg.shape == (1, 64, 64)
        # Background is 0; the three plateaus become classes 1..3.
        assert set(seg.unique().tolist()) == {0, 1, 2, 3}

    def test_labels_are_ordered_by_increasing_intensity(self):
        img = _brain_phantom()
        seg = OtsuTissueSegmenter().segment(img)
        means = [img[0, 0][seg[0] == c].mean() for c in range(1, N_TISSUE_CLASSES + 1)]
        assert means[0] < means[1] < means[2]

    def test_handles_negative_range_unlike_a_0_1_histogram(self):
        """The phantom lives in [-1, 1]; a [0, 1]-hardcoded Otsu would drop the
        entire negative half (this is the latent bug in qa_metrics.hd95)."""
        img = _brain_phantom()
        assert float(img.min()) < 0.0
        mask = OtsuTissueSegmenter().brain_mask(img)
        # The brain is found and is neither empty nor the whole FOV.
        assert 0.0 < float(mask.float().mean()) < 1.0

    def test_constant_image_does_not_crash(self):
        seg = OtsuTissueSegmenter().segment(torch.zeros(1, 1, 16, 16))
        assert seg.shape == (1, 16, 16)


class TestPerfectPredictionScoresPerfectly:
    @pytest.mark.parametrize("name", _SEG_METRICS)
    def test_identical_images(self, name: str):
        metric = MetricsRegistry.get(name, device="cpu")
        img = _brain_phantom()
        value = float(metric(img, img.clone()))

        if name == "tissue_hd95":
            assert value == pytest.approx(0.0, abs=1e-6)
        else:
            assert value == pytest.approx(1.0, abs=1e-6)


class TestDegradationIsPenalised:
    """A metric that cannot tell a degraded prediction from a perfect one is a facade."""

    def test_blur_lowers_dice_and_raises_boundary_distance(self):
        target = _brain_phantom()
        blurred = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(target, (4, 4, 4, 4), mode="replicate"),
            kernel_size=9,
            stride=1,
        )

        dice = MetricsRegistry.get("tissue_dice", device="cpu")
        hd95 = MetricsRegistry.get("tissue_hd95", device="cpu")

        assert float(dice(blurred, target)) < float(dice(target, target.clone()))
        assert float(hd95(blurred, target)) > 0.0

    def test_volume_similarity_penalises_a_size_error(self):
        target = _brain_phantom(shift=0)
        grown = _brain_phantom(shift=6)  # same structure, wrong tissue volumes
        vol = MetricsRegistry.get("tissue_volume_similarity", device="cpu")

        assert float(vol(grown, target)) < float(vol(target, target.clone()))

    def test_brain_mask_dice_penalises_a_shrunken_head(self):
        target = _brain_phantom(shift=0)
        shrunk = _brain_phantom(shift=-10)
        mask_dice = MetricsRegistry.get("brain_mask_dice", device="cpu")

        assert float(mask_dice(shrunk, target)) < 1.0

    def test_dice_does_not_credit_a_collapsed_prediction(self):
        """A constant (DC-blob) output must not score well."""
        target = _brain_phantom()
        collapsed = torch.zeros_like(target)
        dice = MetricsRegistry.get("tissue_dice", device="cpu")

        assert float(dice(collapsed, target)) < 0.5

    def test_hd95_does_not_credit_a_collapsed_prediction(self):
        """Regression: MONAI returns nan for every class a collapsed prediction has
        lost. Averaging over only the finite entries handed that prediction a
        distance of 0.0 — a PERFECT score for total collapse, i.e. the very facade
        this metric exists to catch. A class present in the target but missing from
        the prediction must score the FOV diagonal instead."""
        target = _brain_phantom(size=64)
        collapsed = torch.full_like(target, -1.0)  # blank: no tissue classes at all
        hd95 = MetricsRegistry.get("tissue_hd95", device="cpu")

        value = float(hd95(collapsed, target))
        diagonal = (64.0**2 + 64.0**2) ** 0.5

        assert value == pytest.approx(
            diagonal, rel=1e-6
        ), "a collapsed prediction must score the worst possible boundary distance"
        # And it must be far worse than a merely blurred prediction.
        blurred = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(target, (4, 4, 4, 4), mode="replicate"),
            kernel_size=9,
            stride=1,
        )
        assert float(hd95(blurred, target)) < value


class TestNeverReturnsNaN:
    """train.py sums validation metrics with no non-finite guard — one NaN poisons
    the metric for the entire evaluation pass."""

    @pytest.mark.parametrize("name", _SEG_METRICS)
    @pytest.mark.parametrize(
        "pred,target",
        [
            (torch.full((1, 1, 32, 32), -1.0), torch.full((1, 1, 32, 32), -1.0)),
            (torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32)),
            (_brain_phantom(), torch.full((1, 1, 64, 64), -1.0)),
            (torch.full((1, 1, 64, 64), -1.0), _brain_phantom()),
        ],
        ids=["empty-empty", "constant-constant", "brain-vs-empty", "empty-vs-brain"],
    )
    def test_degenerate_inputs_are_finite(self, name, pred, target):
        value = float(MetricsRegistry.get(name, device="cpu")(pred, target))
        assert torch.isfinite(torch.tensor(value)), f"{name} returned {value}"

    @pytest.mark.parametrize("name", _SEG_METRICS)
    def test_empty_target_is_excluded_not_scored(self, name):
        """An all-background target has no anatomy to grade, so it must fall back to
        the declared degenerate value rather than emit NaN."""
        empty = torch.full((1, 1, 32, 32), -1.0)
        value = float(MetricsRegistry.get(name, device="cpu")(empty, empty))
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_foreground_guard_drops_empty_slices_from_the_batch_mean(self):
        """A batch of [real slice, background slice] must score the real slice only —
        the background slice would otherwise drag a perfect Dice down."""
        brain = _brain_phantom()
        empty = torch.full((1, 1, 64, 64), -1.0)
        pred = torch.cat([brain, empty], dim=0)
        target = torch.cat([brain.clone(), empty.clone()], dim=0)

        # Slice 1 is below the guard: it must not be averaged in.
        assert float(empty.gt(-1.0).float().mean()) < MIN_FOREGROUND_FRACTION
        value = float(MetricsRegistry.get("tissue_dice", device="cpu")(pred, target))
        assert value == pytest.approx(1.0, abs=1e-6)


class TestVolumetricInputs:
    """The slab arms (stage1_sparse_vae_slab / stage1_slat_slab_to_volume) are
    spatial_dims=3 and keep the depth axis, so these metrics are handed a 5-D
    [B, C, H, W, D] tensor. They must handle it rather than silently mis-shape."""

    @staticmethod
    def _volume(shift: int = 0, depth: int = 3) -> torch.Tensor:
        slab = torch.cat([_brain_phantom(shift=shift) for _ in range(depth)], dim=0)
        # [D, 1, H, W] -> [1, 1, H, W, D]
        return slab.permute(1, 2, 3, 0).unsqueeze(0)

    def test_segmenter_keeps_the_depth_axis(self):
        vol = self._volume()
        seg = OtsuTissueSegmenter().segment(vol)
        assert seg.shape == (1, 64, 64, 3)

    @pytest.mark.parametrize("name", _SEG_METRICS)
    def test_metrics_run_on_5d_and_score_a_perfect_match(self, name: str):
        vol = self._volume()
        value = float(MetricsRegistry.get(name, device="cpu")(vol, vol.clone()))
        expected = 0.0 if name == "tissue_hd95" else 1.0
        assert value == pytest.approx(expected, abs=1e-6)

    def test_5d_degradation_is_penalised(self):
        target = self._volume(shift=0)
        grown = self._volume(shift=6)
        dice = MetricsRegistry.get("tissue_dice", device="cpu")
        assert float(dice(grown, target)) < 1.0


class TestRegistryContract:
    @pytest.mark.parametrize("name", _SEG_METRICS)
    def test_registered_and_declares_direction(self, name: str):
        assert MetricsRegistry.is_registered(name)
        metric = MetricsRegistry.get(name, device="cpu")
        assert isinstance(metric.higher_is_better, bool)

    def test_directions_are_what_the_science_expects(self):
        def get(n: str) -> bool:
            return MetricsRegistry.get(n, device="cpu").higher_is_better

        assert get("tissue_dice") is True
        assert get("brain_mask_dice") is True
        assert get("tissue_volume_similarity") is True
        assert get("tissue_hd95") is False  # a distance: lower is better


class TestOutlierRobustHistogram:
    """Regression: a hot voxel used to erase the brain mask entirely.

    ``_otsu_thresholds`` spanned its histogram over ``[min, max]``. MRI coil
    combination leaves isolated hot voxels, and the sim2rank pseudo-GT path
    normalises by p99, so ``max`` is literally the outlier ratio -- the
    2026-07-25 fastMRI-brain sweep logged max/p99 from 1.5 to 1297 (median
    101). At 64 bins a ratio of ~300 puts every voxel of anatomy in bin 0,
    Otsu can no longer resolve tissue, the mask comes back empty, and all
    four metrics fall through to their degenerate constant. That run emitted
    5,200 "no anatomy to grade" warnings -- 1,300 per metric, i.e. every
    single call -- and reported 0.0 for the whole sweep as if it were a
    measurement.
    """

    @staticmethod
    def _with_hot_voxel(ratio: float) -> torch.Tensor:
        """Phantom rescaled to [0, 1], then given one voxel at *ratio*."""
        img = (_brain_phantom() + 1.0) / 2.0
        img[0, 0, 0, 0] = ratio
        return img

    @pytest.mark.parametrize("ratio", [10.0, 100.0, 299.0, 1297.0])
    def test_hot_voxel_does_not_erase_the_brain_mask(self, ratio: float):
        segmenter = OtsuTissueSegmenter()
        mask = segmenter.brain_mask(self._with_hot_voxel(ratio))
        frac = float(mask.float().mean())

        assert frac > MIN_FOREGROUND_FRACTION, (
            f"max/p99={ratio:g} collapsed the brain mask to {frac:.4f}; the Otsu "
            "histogram is being set by the outlier instead of the anatomy"
        )

    def test_mask_is_stable_across_the_observed_outlier_range(self):
        """The mask must not *drift* with the outlier either, only survive it."""
        segmenter = OtsuTissueSegmenter()
        baseline = float(
            segmenter.brain_mask((_brain_phantom() + 1.0) / 2.0).float().mean()
        )
        for ratio in (10.0, 100.0, 299.0, 1297.0):
            frac = float(segmenter.brain_mask(self._with_hot_voxel(ratio)).float().mean())
            assert frac == pytest.approx(baseline, abs=0.02), (
                f"brain mask moved from {baseline:.4f} to {frac:.4f} at ratio {ratio:g}"
            )

    def test_tissue_classes_survive_the_outlier(self):
        """Not just the mask: the 3-class partition must still be resolved."""
        segmenter = OtsuTissueSegmenter()
        labels = segmenter.segment(self._with_hot_voxel(299.0))
        assert set(labels.unique().tolist()) == set(range(N_TISSUE_CLASSES + 1))

    @pytest.mark.parametrize("name", _SEG_METRICS)
    def test_metrics_still_measure_under_an_outlier(self, name: str):
        """The end-to-end symptom: a degraded pred must still score worse."""
        metric = MetricsRegistry.get(name, device="cpu")
        target = self._with_hot_voxel(299.0)
        degraded = self._with_hot_voxel(299.0)
        degraded[..., :20, :] = 0.0  # remove a slab of anatomy

        good = float(metric(target, target))
        bad = float(metric(degraded, target))

        assert good != pytest.approx(bad), (
            f"{name} returned {good} for both a perfect and a mutilated "
            "prediction -- it is reporting the degenerate constant, not measuring"
        )

    def test_clip_is_a_noop_on_a_well_scaled_image(self):
        """Guard against the fix silently changing healthy-image behaviour."""
        segmenter = OtsuTissueSegmenter()
        clean = (_brain_phantom() + 1.0) / 2.0
        frac = float(segmenter.brain_mask(clean).float().mean())
        assert 0.1 < frac < 0.6  # the phantom's structure occupies ~1/4 of the FOV
