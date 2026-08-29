"""Temporal metrics for the time-series regimes (``mri_functional``, ``mri_dynamic``).

Both regimes reconstruct a **series**, and both are graded by default on
per-frame PSNR/SSIM — which cannot see the failure that matters most to either.
A reconstruction that emits the temporal *mean* at every frame scores excellent
per-frame PSNR while having destroyed every dynamic in the data: the frames are
individually plausible and collectively meaningless. Per-frame metrics average
over exactly the axis the claim lives on, so no amount of them detects it.

That is why these are regime-tagged rather than agnostic: they measure along the
time axis, which is what makes a functional or dynamic acquisition what it is.

Conventions shared by both metrics here:

* Input is ``[B, T, H, W]`` (or ``[B, T, ...]``) with **time on axis 1** — the
  layout the dynamic/fMRI datasets and strategies use, where the T frames ride
  the channel axis.
* Real-valued input only. A complex tensor is an un-reconstructed image, not a
  time series, so it returns ``NaN`` (skip) rather than silently grading
  ``abs()`` — the guard ``b0_field_rmse`` and ``adc_mae`` established.
* Degenerate input returns ``NaN`` (skip), never a sentinel in the metric's
  ordinary range. A number that looks like a measurement but is not one is worse
  than no number (pitfall #9).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from mriforge.config.schemas.enums import Regime, Task
from mriforge.core.metrics.registry import register_metric

logger = logging.getLogger(__name__)

#: Below this, a voxel's time-course is noise and its correlation is arbitrary.
_MIN_TEMPORAL_STD = 1e-6


def _otsu_threshold(values: torch.Tensor, bins: int = 256) -> float | None:
    """Otsu's bimodal split of ``values`` (1-D), or ``None`` if it is unimodal.

    Chosen over a fixed intensity percentile because a percentile encodes an
    assumption about *how much* of the image is background, and that assumption
    is what broke ``TemporalSNR``: a 25th-percentile threshold only excludes air
    if air occupies under 25% of the FOV, while a real EPI FOV is typically
    60-80% air. Otsu instead finds the split that maximises between-class
    variance, so it adapts to the actual air fraction rather than presuming one.

    Reference: Otsu, IEEE Trans Syst Man Cybern 9(1):62-66, 1979.
    """
    lo, hi = float(values.min()), float(values.max())
    if not (hi > lo):
        return None
    hist = torch.histc(values, bins=bins, min=lo, max=hi)
    total = hist.sum()
    if float(total) <= 0:
        return None
    prob = hist / total
    centers = torch.linspace(lo, hi, bins, device=values.device, dtype=values.dtype)

    omega = torch.cumsum(prob, dim=0)  # class-0 mass at each split
    mu = torch.cumsum(prob * centers, dim=0)  # class-0 first moment
    mu_total = mu[-1]
    # Between-class variance; undefined where a class is empty.
    denom = omega * (1.0 - omega)
    numer = (mu_total * omega - mu) ** 2
    sigma_b2 = torch.where(
        denom > 0,
        numer / denom.clamp_min(torch.finfo(values.dtype).tiny),
        torch.zeros_like(numer),
    )
    if float(sigma_b2.max()) <= 0:
        return None  # unimodal — no honest foreground/background split
    return float(centers[int(sigma_b2.argmax())])


def _reject_non_series(series: torch.Tensor | None, label: str) -> str | None:
    """Return a skip-reason if ``series`` is not a usable real time series."""
    if series is None:
        return f"{label} is None"
    if series.is_complex():
        return f"{label} is complex — an un-reconstructed image, not a time series"
    if series.ndim < 3:
        return f"{label} must be [B, T, ...] with time on axis 1, got {tuple(series.shape)}"
    if series.shape[1] < 2:
        return (
            f"{label} has {series.shape[1]} frame(s) on axis 1 — a single frame "
            "has no temporal axis to measure"
        )
    return None


@register_metric(
    "tsnr",
    aliases=["temporal_snr", "TemporalSNR"],
    direction="higher",
    workflows=frozenset({Regime.FUNCTIONAL}),
)
class TemporalSNR:
    r"""Temporal signal-to-noise ratio of a BOLD series: ``mean_t(S) / std_t(S)``.

    The standard fMRI quality IQM (Triantafyllou et al., NeuroImage 26(1):243-250,
    2005; reported by MRIQC). Computed per voxel after removing a linear trend,
    then averaged over **foreground** voxels — air has ``mean_t ~ 0``, so an
    arbitrary ratio there would dilute the average toward a number describing
    nothing.

    Pass an explicit ``mask`` when one exists; that is what MRIQC does and it is
    the only way to be exact. Without one the foreground is split by **Otsu's
    method**, which adapts to whatever fraction of the FOV is air.

    A fixed intensity percentile is NOT usable here, and the 25th percentile this
    originally used was actively wrong: it excludes air only when air occupies
    **less than 25% of the FOV**, and a real EPI FOV is typically 60-80% air. The
    threshold then landed inside air, most air voxels passed, and the score
    tracked FOV/cropping rather than image quality — at a realistic 30% brain
    fraction it under-reported by **2.4x** (~41 against a true ~100). The failure
    was invisible on a synthetic mostly-brain fixture, which is exactly the shape
    the tests here now guard against.

    Tagged ``mri_functional``: tSNR presupposes a repeated acquisition of a
    *static* object, where all temporal variance beyond the drift is noise. That
    is true of a BOLD run and false of a cine or a DCE series, where the temporal
    variance is the signal — hence no ``mri_dynamic`` or ``mri_perfusion`` tag.

    **Read it with the known caveat**: tSNR rewards temporal smoothing. A method
    that low-pass filters along time raises tSNR without recovering anything — it
    measures noise, not fidelity, and cannot tell them apart. Pair it with
    :class:`TemporalFidelity`, which fails exactly where tSNR is fooled. This is
    a property of tSNR everywhere, not of this implementation.

    The total collapse (emitting the temporal mean at every frame) is the one
    case tSNR does *not* reward: with zero temporal variance every voxel fails
    the foreground guard and the result is ``NaN`` (skip), not infinity. The
    dangerous case is the partial smoother, which damages the dynamics and still
    scores better — pinned by
    ``test_tsnr_rewards_temporal_smoothing_the_documented_caveat``.

    Reference-free: ``target`` is accepted and ignored (tSNR is computed from the
    prediction alone), which is why it is registered ``requires_reference=False``
    at the schema layer.
    """

    @property
    def name(self) -> str:
        return "tsnr"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> float:
        del target, kwargs  # tSNR is reference-free by construction.
        reason = _reject_non_series(prediction, "prediction")
        if reason is not None:
            logger.debug("tsnr skipped: %s.", reason)
            return float("nan")

        series = prediction.float()
        n_t = series.shape[1]

        # Remove a linear drift before measuring noise. Scanner drift is a
        # low-frequency artifact, not noise; leaving it in inflates std_t and
        # understates tSNR — every fMRI QA pipeline detrends first.
        t = torch.linspace(-1.0, 1.0, n_t, device=series.device, dtype=series.dtype)
        t = t.reshape(1, n_t, *([1] * (series.ndim - 2)))
        t_centred = t - t.mean()
        denom = (t_centred**2).sum()
        slope = (series * t_centred).sum(dim=1, keepdim=True) / denom
        residual = series - slope * t_centred  # drift removed, mean retained

        mean_t = residual.mean(dim=1)
        # Unbiased std needs T-1 >= 1, guaranteed by the min_frames=2 guard.
        std_t = residual.std(dim=1)

        # Foreground only: in air mean_t ~ 0 and the ratio is arbitrary noise,
        # which would drag the average toward a number that describes nothing.
        if mask is not None:
            if mask.shape != mean_t.shape:
                logger.debug(
                    "tsnr skipped: mask %s does not match the per-voxel map %s.",
                    tuple(mask.shape),
                    tuple(mean_t.shape),
                )
                return float("nan")
            foreground = mask.to(torch.bool)
        else:
            threshold = _otsu_threshold(mean_t.flatten().float())
            if threshold is None:
                logger.debug(
                    "tsnr skipped: the temporal mean is unimodal, so there is no "
                    "foreground to separate from background."
                )
                return float("nan")
            foreground = mean_t > threshold

        foreground = foreground & (std_t > _MIN_TEMPORAL_STD)
        if not bool(foreground.any()):
            logger.debug("tsnr skipped: no foreground voxel has temporal variance.")
            return float("nan")

        return float((mean_t[foreground] / std_t[foreground]).mean())


@register_metric(
    "temporal_fidelity",
    aliases=["temporal_correlation", "TemporalFidelity"],
    direction="higher",
    workflows=frozenset({Regime.FUNCTIONAL, Regime.DYNAMIC}),
    tasks=frozenset({Task.RECONSTRUCTION, Task.DENOISING, Task.SUPER_RESOLUTION}),
)
class TemporalFidelity:
    r"""Mean Pearson correlation between predicted and reference voxel time-courses.

    For each voxel, correlate the predicted time-course against the reference's,
    and average over voxels whose reference course actually varies. In ``[-1, 1]``;
    higher is better; ``1`` means every voxel's dynamics are reproduced up to an
    affine rescaling.

    **This is the metric that catches temporal-mean collapse.** A reconstruction
    that outputs the temporal mean at every frame scores high per-frame PSNR and
    SSIM — each frame is a perfectly good image of the average — while having
    thrown away every dynamic in the acquisition. Per-frame metrics reduce over
    the time axis, so the collapse is invisible to them by construction. Under
    this metric that reconstruction has zero temporal variance, correlates with
    nothing, and scores ``NaN``/0.

    Being correlation-based, it is deliberately blind to per-voxel scale and
    offset: it asks whether the *shape* of the dynamics survived, not whether the
    intensities are calibrated. PSNR already answers the latter, so pairing the
    two covers both, and neither alone does.

    Tagged ``{mri_functional, mri_dynamic}`` — the two regimes whose **output is
    a time series**. Not ``mri_perfusion``: a DCE acquisition is temporal too, but
    its output is a parameter map, and its time-courses are graded by the Tofts
    refit rather than correlated frame-by-frame.

    Voxels whose *reference* course is flat (``std < 1e-6``) are excluded rather
    than counted: correlation is undefined at zero variance, and scoring them 0
    would penalise a perfect reconstruction of a genuinely static background.
    """

    @property
    def name(self) -> str:
        return "temporal_fidelity"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        del kwargs
        reason = _reject_non_series(prediction, "prediction") or _reject_non_series(
            target, "target"
        )
        if reason is None and prediction.shape != target.shape:
            reason = f"shape mismatch {tuple(prediction.shape)} vs {tuple(target.shape)}"
        if reason is not None:
            logger.debug("temporal_fidelity skipped: %s.", reason)
            return float("nan")

        pred = prediction.float()
        ref = target.float()

        pred_c = pred - pred.mean(dim=1, keepdim=True)
        ref_c = ref - ref.mean(dim=1, keepdim=True)

        # Both gates below are on a per-voxel STD, and that is load-bearing.
        #
        # They were previously a mismatched pair — gradeability on `ref_norm`,
        # scoring on `pred_norm * ref_norm`, against the same constant. For a
        # perfect reconstruction the product is `ref_norm**2`, so scoring 1.0
        # silently required `ref_norm > 1e-3` while gradeability required only
        # `> 1e-6`. Every voxel in between was graded and scored 0.0 despite
        # being BIT-PERFECT — which is exactly the value this metric reserves for
        # total temporal collapse, on a higher-is-better metric. The two opposite
        # verdicts were indistinguishable.
        #
        # A std is also T-independent, where a norm is not: `norm = std*sqrt(T-1)`,
        # so gating a norm on `_MIN_TEMPORAL_STD` made the effective threshold
        # drift with the number of frames.
        pred_std = pred_c.pow(2).mean(dim=1).sqrt()
        ref_std = ref_c.pow(2).mean(dim=1).sqrt()

        # Only voxels whose REFERENCE varies are gradeable. A voxel that is
        # static in the reference has no dynamics to reproduce, so scoring it
        # would measure the background, not the method.
        gradeable = ref_std > _MIN_TEMPORAL_STD
        if not bool(gradeable.any()):
            logger.debug(
                "temporal_fidelity skipped: the reference series has no temporal "
                "variance anywhere — there are no dynamics to grade."
            )
            return float("nan")

        # A prediction that is flat where the reference varies is the collapse
        # this metric exists to catch. Its correlation is 0/0; force it to 0.0
        # (no agreement) rather than dropping the voxel, which would silently
        # EXCLUDE the failure and average only over the voxels that worked.
        numerator = (pred_c * ref_c).sum(dim=1)
        denominator = pred_c.pow(2).sum(dim=1).sqrt() * ref_c.pow(2).sum(dim=1).sqrt()
        # The clamp only guards the division itself; `tiny` is ~1e-38 in float32,
        # so it can never mask a real voxel the way a 1e-6 clamp did.
        corr = torch.where(
            pred_std > _MIN_TEMPORAL_STD,
            numerator / denominator.clamp_min(torch.finfo(pred.dtype).tiny),
            torch.zeros_like(numerator),
        )
        return float(corr[gradeable].mean())


__all__ = ["TemporalFidelity", "TemporalSNR"]
