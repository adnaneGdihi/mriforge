"""Temporal metrics for mri_functional / mri_dynamic.

The load-bearing test is ``test_temporal_mean_collapse_...``: it demonstrates the
failure that per-frame metrics structurally cannot see, and which is the entire
reason these metrics are regime-tagged.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.config.schemas.enums import Regime
from spectramr.core.metrics.temporal_metrics import TemporalFidelity, TemporalSNR


def _series(batch: int = 2, frames: int = 20, size: int = 6) -> torch.Tensor:
    """A [B, T, H, W] series whose voxels genuinely vary over time."""
    t = torch.linspace(0, 4 * math.pi, frames).reshape(1, frames, 1, 1)
    phase = torch.rand(batch, 1, size, size) * math.pi
    return 10.0 + torch.sin(t + phase)


def _brain_in_air(
    brain_fraction: float,
    *,
    frames: int = 40,
    size: int = 20,
    noise: float = 1.0,
    level: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A constant-signal "brain" surrounded by air, plus its exact mask.

    The brain's true tSNR is ``level / noise`` (100 by default) and does NOT
    depend on ``brain_fraction`` — only the amount of surrounding air does. Any
    tSNR worth the name must therefore return the same number for every
    fraction; that is precisely what the percentile threshold got wrong.
    """
    torch.manual_seed(0)
    base = torch.zeros(1, frames, size, size)
    n_brain = int(size * size * brain_fraction)
    base.view(1, frames, -1)[:, :, :n_brain] = level
    mask = torch.zeros(1, size, size, dtype=torch.bool)
    mask.view(1, -1)[:, :n_brain] = True
    return base + torch.randn_like(base) * noise, mask


class TestTemporalFidelity:
    def test_perfect_reconstruction_scores_one(self) -> None:
        ref = _series()
        assert TemporalFidelity()(ref.clone(), ref) == pytest.approx(1.0, abs=1e-5)

    def test_temporal_mean_collapse_is_caught_where_per_frame_metrics_cannot_see_it(
        self,
    ) -> None:
        """THE test. A recon emitting the temporal mean at every frame.

        Every frame is a perfectly good image of the average, so per-frame PSNR
        stays high — the dynamics are gone but per-frame metrics reduce over the
        time axis and cannot detect it. temporal_fidelity is the metric that
        does: with zero temporal variance the prediction correlates with nothing.
        """
        ref = _series()
        collapsed = ref.mean(dim=1, keepdim=True).expand_as(ref).contiguous()

        # Per-frame PSNR still looks respectable — the collapse is invisible.
        mse = float(((collapsed - ref) ** 2).mean())
        psnr = 10 * math.log10(float(ref.max()) ** 2 / mse)
        assert psnr > 20, "precondition: the collapse must LOOK fine to PSNR"

        # ...and temporal fidelity sees straight through it.
        assert TemporalFidelity()(collapsed, ref) == pytest.approx(0.0, abs=1e-5)

    def test_collapse_is_scored_zero_not_silently_excluded(self) -> None:
        """A flat prediction must count as failure, not be dropped from the mean.

        Excluding zero-variance PREDICTION voxels would average only over the
        voxels that happened to work — the failure would vanish from the number
        instead of being reported by it.
        """
        ref = _series(batch=1, frames=10, size=2)
        half = ref.clone()
        half[:, :, 0, :] = ref[:, :, 0, :].mean(dim=1, keepdim=True)  # collapse half
        score = TemporalFidelity()(half, ref)
        assert 0.2 < score < 0.8, f"expected a partial score, got {score}"

    def test_static_reference_voxels_are_excluded_not_penalised(self) -> None:
        """A perfect recon of a static background must not be marked down.

        Correlation is undefined at zero reference variance; scoring those voxels
        0 would punish a flawless reconstruction of genuinely static tissue.
        """
        ref = _series(batch=1, frames=10, size=4)
        ref[:, :, 0, 0] = 5.0  # a perfectly static voxel
        assert TemporalFidelity()(ref.clone(), ref) == pytest.approx(1.0, abs=1e-5)

    def test_insensitive_to_per_voxel_scale_and_offset(self) -> None:
        """Correlation grades the SHAPE of the dynamics; PSNR grades calibration."""
        ref = _series()
        rescaled = ref * 3.0 + 7.0
        assert TemporalFidelity()(rescaled, ref) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("scale", [1.0, 1e-2, 1e-3, 5e-4, 1e-4])
    def test_a_perfect_reconstruction_scores_one_at_any_series_scale(
        self, scale: float
    ) -> None:
        """RED-against-the-bug: scaling DOWN, which is where it broke.

        The gradeable gate tested `ref_norm > 1e-6` while the scoring gate tested
        `pred_norm * ref_norm > 1e-6` against the same constant. For a perfect
        reconstruction the product is `ref_norm**2`, so scoring 1.0 silently
        required `ref_norm > 1e-3` — every voxel in between was graded and scored
        **0.0 despite being bit-perfect**, which is exactly the value this metric
        reserves for total temporal collapse. On a higher-is-better metric the two
        opposite verdicts were indistinguishable.

        `test_insensitive_to_per_voxel_scale_and_offset` above asserts this very
        property and could not catch it: `ref * 3.0 + 7.0` scales *up*, away from
        the threshold. Only the down direction reaches it.
        """
        ref = _series() * scale
        assert TemporalFidelity()(ref.clone(), ref) == pytest.approx(1.0, abs=1e-4)

    def test_an_ungradeable_series_skips_rather_than_reporting_collapse(self) -> None:
        """Below the variance floor the answer is NaN (skip), never 0.0.

        0.0 means "the dynamics were destroyed". A series with no measurable
        dynamics to begin with has not been destroyed — it is unmeasurable, and
        saying so is the honest answer.
        """
        ref = _series() * 1e-9
        assert math.isnan(TemporalFidelity()(ref.clone(), ref))

    @pytest.mark.parametrize("n_frames", [10, 40, 160])
    def test_the_variance_gate_is_independent_of_frame_count(
        self, n_frames: int
    ) -> None:
        """The gate is on a std, not a norm: `norm = std*sqrt(T-1)`.

        Gating a norm on a std-scaled constant made the effective threshold drift
        with T, so the same physical series became gradeable or not depending only
        on how many frames were acquired.
        """
        t = torch.linspace(0, 4 * math.pi, n_frames).reshape(1, n_frames, 1, 1)
        ref = (10.0 + torch.sin(t + torch.rand(1, 1, 4, 4))) * 1e-4
        assert TemporalFidelity()(ref.clone(), ref) == pytest.approx(1.0, abs=1e-4)

    def test_anticorrelated_prediction_scores_negative(self) -> None:
        ref = _series()
        flipped = -ref
        assert TemporalFidelity()(flipped, ref) == pytest.approx(-1.0, abs=1e-5)

    def test_fully_static_reference_skips(self) -> None:
        ref = torch.full((1, 8, 3, 3), 4.0)
        assert math.isnan(TemporalFidelity()(ref.clone(), ref))

    @pytest.mark.parametrize(
        "bad",
        [
            torch.randn(2, 1, 4, 4),  # single frame: no temporal axis
            torch.randn(2, 4),  # not [B, T, ...]
        ],
    )
    def test_non_series_input_skips(self, bad: torch.Tensor) -> None:
        assert math.isnan(TemporalFidelity()(bad, bad.clone()))

    def test_complex_input_skips(self) -> None:
        """A complex tensor is an un-reconstructed image, not a time series."""
        c = torch.randn(1, 8, 3, 3, dtype=torch.complex64)
        assert math.isnan(TemporalFidelity()(c, c.clone()))

    def test_shape_mismatch_skips(self) -> None:
        assert math.isnan(TemporalFidelity()(_series(frames=10), _series(frames=8)))

    def test_tagged_for_both_time_series_regimes(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        tags = MetricsRegistry._workflow_tags["temporal_fidelity"]["workflows"]
        assert tags == frozenset({Regime.FUNCTIONAL, Regime.DYNAMIC})


class TestTemporalSNR:
    def test_less_noise_scores_higher(self) -> None:
        base = torch.full((1, 40, 5, 5), 100.0)
        torch.manual_seed(0)
        noisy = base + torch.randn_like(base) * 5.0
        clean = base + torch.randn_like(base) * 1.0
        assert TemporalSNR()(clean) > TemporalSNR()(noisy)

    def test_linear_drift_is_removed_before_measuring_noise(self) -> None:
        """Drift is a low-frequency artifact, not noise.

        Leaving it in inflates std_t and understates tSNR, which is why every
        fMRI QA pipeline detrends first.
        """
        torch.manual_seed(0)
        n_t = 40
        noise = torch.randn(1, n_t, 4, 4) * 1.0
        flat = 100.0 + noise
        drift = torch.linspace(0, 30, n_t).reshape(1, n_t, 1, 1)
        drifting = flat + drift

        # Same noise, plus a pure linear trend => the same tSNR after detrending.
        assert TemporalSNR()(drifting) == pytest.approx(
            float(TemporalSNR()(flat)), rel=0.15
        )

    def test_it_is_reference_free(self) -> None:
        """tSNR is computed from the prediction alone; target must be ignored."""
        series = _series()
        assert TemporalSNR()(series) == pytest.approx(
            float(TemporalSNR()(series, torch.randn_like(series))), rel=1e-6
        )

    def test_tsnr_rewards_temporal_smoothing_the_documented_caveat(self) -> None:
        """Pins the weakness the docstring warns about, so it stays documented.

        A method that low-passes along time raises tSNR *without recovering
        anything* — it measures noise, not fidelity, and cannot tell the two
        apart. This is a property of tSNR everywhere, not of this
        implementation. TemporalFidelity is the metric that catches what tSNR
        misses, which is why the docstring pairs them.

        Two details are load-bearing and were both wrong on the first attempt:

        * The signal must have FAST dynamics (here ~4 frames per cycle, near the
          temporal Nyquist). Against a slow sinusoid a moving average is a
          genuinely good denoiser and improves fidelity too — there is no trap to
          demonstrate, because smoothing only destroys dynamics faster than its
          own kernel.
        * Smoothing must be a real moving average, not a total collapse: a
          perfectly flat series has exactly zero temporal std, every voxel fails
          the foreground guard, and tSNR skips to NaN rather than being fooled.
          The dangerous case is the partial smoother.
        """
        torch.manual_seed(0)
        n_t = 40
        t = torch.linspace(0, 2 * math.pi, n_t).reshape(1, n_t, 1, 1)
        phase = torch.rand(1, 1, 4, 4) * math.pi
        signal = 10.0 + torch.sin(t * 10 + phase)  # ~4 frames/cycle
        noisy = signal + torch.randn_like(signal) * 0.5

        # A 5-tap temporal moving average: strictly less noise, blurred dynamics.
        # Replicate-pad, NOT zero-pad: zeros at the ends would fabricate a huge
        # step at frames 0..1 and T-2..T-1, inflating std_t and making the
        # smoother look WORSE for a reason that has nothing to do with tSNR.
        kernel = torch.ones(1, 1, 5) / 5.0
        b, t, h, w = noisy.shape
        flat = noisy.permute(0, 2, 3, 1).reshape(-1, 1, t)
        padded = torch.nn.functional.pad(flat, (2, 2), mode="replicate")
        smoothed = (
            torch.nn.functional.conv1d(padded, kernel)
            .reshape(b, h, w, t)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Smoothing DESTROYED the dynamics — fidelity goes negative, i.e. the
        # smoothed course is anticorrelated with the truth (0.82 -> -0.45).
        assert TemporalFidelity()(smoothed, signal) < 0.0
        assert TemporalFidelity()(noisy, signal) > 0.7
        # ...and tSNR rewarded it lavishly anyway: 11.6 -> 39.8. That is the
        # trap, and the reason tsnr must never be read as a fidelity score.
        assert TemporalSNR()(smoothed) > 2.0 * TemporalSNR()(noisy)

    def test_a_totally_collapsed_series_skips_rather_than_reporting_infinity(
        self,
    ) -> None:
        """Zero temporal variance everywhere means tSNR is not measurable.

        mean/std would be a division by zero. Returning NaN (skip) says "not
        computed"; reporting inf, or a large finite number, would rank a dead
        reconstruction top of a leaderboard.
        """
        flat = torch.full((1, 20, 4, 4), 10.0)
        assert math.isnan(TemporalSNR()(flat))

    @pytest.mark.parametrize("brain_fraction", [0.9, 0.5, 0.3, 0.2])
    def test_tsnr_is_independent_of_how_much_of_the_fov_is_air(
        self, brain_fraction: float
    ) -> None:
        """RED-against-the-bug: the same brain, cropped differently.

        The foreground was split at the 25th intensity percentile, which excludes
        air only when air is under 25% of the FOV — while a real EPI FOV is
        60-80% air. The threshold landed inside air, most air voxels passed with
        `mean_t/std_t ~ 0`, and the score tracked FOV/cropping rather than image
        quality: at a realistic 30% brain fraction it read ~41 against a true
        ~100, a 2.4x under-report, and the number was not a tSNR in the
        Triantafyllou/MRIQC sense at all.

        Otsu's split adapts to the actual air fraction, so the SAME brain scores
        the same tSNR however much air surrounds it. Only a synthetic
        mostly-brain fixture hides this — hence the parametrisation.
        """
        series, _ = _brain_in_air(brain_fraction)
        assert TemporalSNR()(series) == pytest.approx(100.0, rel=0.15)

    def test_an_explicit_mask_is_exact_where_otsu_only_approximates(self) -> None:
        """Otsu is a heuristic and degrades at extreme air fractions; a mask does not.

        This pins the LIMIT of the fallback rather than pretending it has none.
        At 10% brain the Otsu threshold sits ~2 sigma above the air noise, so ~2%
        of air voxels leak through — and with 9x more air than brain that is
        enough to pull ~100 down to ~83. Otsu is still doing its job (the 25th
        percentile scored ~28 here); it simply cannot be exact when one class
        dwarfs the other.

        That is why `mask` exists and why the docstring says to pass one. 10% is
        past the realistic EPI range (brain is typically 20-40% of the FOV), which
        the parametrised test above covers exactly.
        """
        series, mask = _brain_in_air(0.1)
        assert TemporalSNR()(series, mask=mask) == pytest.approx(100.0, rel=0.08)
        otsu_only = TemporalSNR()(series)
        assert otsu_only < 0.9 * TemporalSNR()(series, mask=mask)

    def test_a_mask_shaped_wrong_skips_rather_than_broadcasting(self) -> None:
        series, _ = _brain_in_air(0.3)
        bad = torch.ones(1, 3, 3, dtype=torch.bool)
        assert math.isnan(TemporalSNR()(series, mask=bad))

    def test_tsnr_still_ranks_image_quality_at_a_fixed_fov(self) -> None:
        """The fix must not cost the metric its actual job."""
        clean, _ = _brain_in_air(0.3, noise=0.5)
        noisy, _ = _brain_in_air(0.3, noise=4.0)
        assert TemporalSNR()(clean) > TemporalSNR()(noisy)

    def test_a_uniformly_bright_low_noise_image_has_a_genuinely_high_tsnr(self) -> None:
        """Not a degenerate case — the honest answer is mean/std, and it is large.

        Written because the first version of this test asserted a skip here, on
        the theory that a uniform image is "unimodal so there is no foreground".
        That was wrong: an all-brain FOV has a real tSNR of 5/0.01 = 500, and
        reporting it is correct. Otsu splits the noise, every voxel is signal, and
        the number means what it says. Kept as the counter-example so nobody
        "fixes" this into a skip.
        """
        uniform = torch.full((1, 20, 8, 8), 5.0) + torch.randn(1, 20, 8, 8) * 0.01
        assert TemporalSNR()(uniform) == pytest.approx(500.0, rel=0.25)

    def test_a_zero_range_image_skips_rather_than_inventing_a_foreground(self) -> None:
        """A perfectly constant image has no split AND no temporal variance.

        Otsu returns None on a zero-range input rather than picking an arbitrary
        bin, so the metric says "not computed" instead of grading a threshold it
        made up.
        """
        from spectramr.core.metrics.temporal_metrics import _otsu_threshold

        assert _otsu_threshold(torch.full((64,), 3.0)) is None
        assert math.isnan(TemporalSNR()(torch.full((1, 20, 8, 8), 3.0)))

    def test_single_frame_skips(self) -> None:
        assert math.isnan(TemporalSNR()(torch.randn(1, 1, 4, 4)))

    def test_complex_input_skips(self) -> None:
        assert math.isnan(TemporalSNR()(torch.randn(1, 8, 3, 3, dtype=torch.complex64)))

    def test_tagged_functional_only(self) -> None:
        """NOT dynamic or perfusion: tSNR presupposes a STATIC object.

        It treats all temporal variance beyond drift as noise — true of a BOLD
        run, false of a cine or DCE series where that variance IS the signal.
        """
        from spectramr.core.metrics.registry import MetricsRegistry

        tags = MetricsRegistry._workflow_tags["tsnr"]["workflows"]
        assert tags == frozenset({Regime.FUNCTIONAL})


class TestRegistryDeclaration:
    """The METRIC_SPECS declaration must match what the metrics actually do.

    ``test_metric_registry_health`` feeds every registered metric a synthetic
    ``[1, 1, 64, 64]`` image pair and asserts the result is finite. That tensor
    has ONE frame, so both metrics here are undefined on it and return ``nan``.
    The health test offers exactly three ways to say "that nan is my contract":
    declare a ``MetricContext`` dependency, raise, or be typed
    ``DOMAIN_SPECIFIC`` in ``METRIC_SPECS``.

    Only the third is honest. ``needs=(...)`` names ``MetricContext`` FIELDS and
    is cross-checked against them by the §6 audit — our requirement is a tensor
    LAYOUT ([B, T, H, W], T >= 2), not a context field, so declaring one we never
    read would be pitfall #16. Undeclared, these hard-fail the health gate.
    """

    def test_both_are_declared_domain_specific(self) -> None:
        pytest.importorskip("scripts.sim2rank.metrics_list")  # not in the public export
        from scripts.sim2rank.metrics_list import METRIC_SPECS, MetricType

        by_key = {s.registry_key: s for s in METRIC_SPECS}
        for key in ("tsnr", "temporal_fidelity"):
            assert key in by_key, (
                f"'{key}' is registered but absent from METRIC_SPECS, so "
                f"test_metric_registry_health will feed it an image pair and "
                f"hard-fail on the nan it correctly returns."
            )
            assert by_key[key].metric_type is MetricType.DOMAIN_SPECIFIC, (
                f"'{key}' must be DOMAIN_SPECIFIC — it needs [B, T, H, W] with "
                f"T >= 2, not a (pred, target) image pair."
            )

    @pytest.mark.parametrize("metric_cls", [TemporalSNR, TemporalFidelity])
    def test_nan_on_the_exact_input_the_health_test_feeds(self, metric_cls) -> None:
        """Pin the BEHAVIOUR the DOMAIN_SPECIFIC declaration describes.

        If someone ever makes these finite on a single-frame pair, the
        declaration above becomes a lie and this test says so — rather than
        leaving the two to drift apart silently.
        """
        g = torch.Generator().manual_seed(0)
        clean = torch.rand(1, 1, 64, 64, generator=g)
        noisy = (clean + 0.15 * torch.randn(1, 1, 64, 64, generator=g)).clamp(0, 1)
        assert math.isnan(metric_cls()(clean, noisy))
