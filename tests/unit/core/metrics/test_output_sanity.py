"""Unit tests for the validation-output degeneracy guard.

The thresholds in :mod:`mriforge.core.metrics.output_sanity` are calibrated
against the 2026-07 MRIxFields2026 cohort, so the tests below pin both the
*mechanism* (each degeneracy shape is detected) and the *calibration* (the
measured healthy/degenerate bands stay on their own sides of the limits). If a
future change moves a limit, the band tests fail with the arm names attached.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.output_sanity import (  # noqa: E402
    AIR_LEVEL_LIMIT,
    BLANK_EXCESS_LIMIT,
    MIN_REFERENCE_AIR,
    measure_output_sanity,
)


def _brain(*, size: int = 32, seed: int = 0, radius_sq: float = 0.25):
    """A crude phantom: a bright centred disc on black air, like an MRI slice."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    disc = ((yy**2 + xx**2) < radius_sq).float()
    img = disc * (0.7 + 0.3 * torch.rand(size, size, generator=g))
    return img.view(1, 1, size, size)


def _air_filled(*, size: int = 32, seed: int = 0, amplitude: float = 1.0):
    """The b22/b27/b33 shape: correct anatomy over a noise-filled air region.

    The noise must VARY, not sit at a constant offset — a constant pedestal is
    removed by the per-sample min-max normalisation and would leave
    ``air_level`` at zero. ``amplitude`` controls how bright the speckle is:
    b33_field_bridge keeps it below tissue (air_level 0.28) and so still
    correlates at +0.91, whereas b22 fills the whole range (air_level 0.68).
    """
    g = torch.Generator().manual_seed(seed)
    img = _brain(size=size, seed=seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    air = ((yy**2 + xx**2) >= 0.25).view(1, 1, size, size)
    speckle = amplitude * torch.rand(1, 1, size, size, generator=g)
    return torch.where(air, speckle, img)


class TestHealthyOutput:
    def test_a_faithful_reconstruction_is_not_flagged(self):
        target = _brain(seed=1)
        pred = target + 0.01 * torch.randn_like(target)
        assert measure_output_sanity(pred, target).verdict is None

    def test_the_guard_is_invariant_to_output_scale(self):
        """A model emitting 0..1000 instead of 0..1 must not be flagged."""
        target = _brain(seed=2)
        scaled = measure_output_sanity(target * 1000.0, target)
        assert scaled.verdict is None
        assert scaled.air_level == pytest.approx(
            measure_output_sanity(target, target).air_level, abs=1e-6
        )


class TestDegenerateOutput:
    def test_air_filled_output_is_flagged(self):
        """b22/b27/b33: anatomy present, but the air region is filled."""
        target = _brain(seed=3)
        report = measure_output_sanity(_air_filled(seed=3), target)
        assert report.verdict == "air_filled"
        assert report.air_level > AIR_LEVEL_LIMIT

    def test_a_high_correlation_output_is_still_flagged(self):
        """b33_field_bridge scored the cohort's BEST correlation (+0.91) while
        being unusable — correlation cannot substitute for this guard."""
        target = _brain(seed=9)
        pred = _air_filled(seed=9, amplitude=0.45)  # speckle stays below tissue
        corr = torch.corrcoef(torch.stack([pred.flatten(), target.flatten()]))[0, 1]
        assert corr > 0.5, "fixture must genuinely correlate with the reference"
        assert measure_output_sanity(pred, target).verdict == "air_filled"

    def test_white_out_is_flagged(self):
        """b23_recoverability_vib: saturated everywhere (posterior collapse)."""
        target = _brain(seed=4)
        pred = torch.ones_like(target)
        assert measure_output_sanity(pred, target).verdict == "constant_output"

    def test_blank_output_is_flagged_even_though_its_air_is_perfect(self):
        """b110_ablate_no_anchor: dead output. air_level alone reads as perfect,
        so the blank_excess rule is what must catch it."""
        target = _brain(seed=5, radius_sq=0.6)  # plenty of anatomy to go missing
        pred = torch.zeros_like(target)
        pred[0, 0, 0, 0] = 1.0  # break the constant-output short-circuit
        report = measure_output_sanity(pred, target)
        assert report.air_level < AIR_LEVEL_LIMIT, "air region is (spuriously) clean"
        assert report.verdict == "blank_output"
        assert report.blank_excess > BLANK_EXCESS_LIMIT


class TestCalibrationBands:
    """Pin the measured 2026-07 cohort bands against the shipped limits."""

    # (arm, air_level) — the extremes of each measured band.
    HEALTHY = [("b110_ilvr_anchor", 0.0817), ("b19_anatomy_latent_7t", 0.0738)]
    DEGENERATE = [
        ("b35_field_cfg", 0.2451),
        ("b33_field_bridge", 0.2845),
        ("b110_ablate_pin", 0.3623),
        ("b22_ablate_likelihood", 0.6796),
        ("b23_recoverability_vib", 0.9391),
        ("b23_ablate_tight_bottleneck", 0.9586),
    ]

    @pytest.mark.parametrize("arm,air_level", HEALTHY)
    def test_measured_healthy_arms_stay_below_the_limit(self, arm, air_level):
        assert air_level < AIR_LEVEL_LIMIT, f"{arm} would become a false positive"

    @pytest.mark.parametrize("arm,air_level", DEGENERATE)
    def test_measured_degenerate_arms_stay_above_the_limit(self, arm, air_level):
        assert air_level > AIR_LEVEL_LIMIT, f"{arm} would go undetected"

    def test_the_limit_keeps_a_wide_margin_on_both_sides(self):
        """Guard against a future tweak that shaves the 3x separation to nothing."""
        worst_healthy = max(v for _, v in self.HEALTHY)
        best_degenerate = min(v for _, v in self.DEGENERATE)
        assert worst_healthy < AIR_LEVEL_LIMIT < best_degenerate
        assert best_degenerate / worst_healthy > 2.5


class TestNeverAbortsValidation:
    def test_shape_mismatch_returns_no_verdict(self):
        r = measure_output_sanity(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16))
        assert r.verdict is None

    def test_reference_without_background_is_skipped(self):
        """A tight crop that is all anatomy has no air region to grade.

        Normalisation manufactures a few percent of "air" in ANY image, so this
        also pins that ``MIN_REFERENCE_AIR`` stays above that manufactured
        floor — otherwise the guard fires on crops it cannot judge.
        """
        target = torch.rand(1, 1, 16, 16) * 0.5 + 0.5
        r = measure_output_sanity(torch.rand_like(target), target)
        assert r.verdict is None
        assert "no background to grade" in r.detail
        assert r.reference_air_fraction < MIN_REFERENCE_AIR
