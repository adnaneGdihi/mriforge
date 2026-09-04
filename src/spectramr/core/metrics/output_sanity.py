r"""Validation-output degeneracy guard (pitfall #20, artefact layer).

SSIM/PSNR grade *agreement with the reference*; they do not ask whether the
prediction is a picture of anything. A model can score a high correlation and
still be unusable — ``mrixfields_b33_field_bridge`` reconstructs correct
anatomy (best-tile correlation ``+0.91``, the highest in its cohort) on top of
a grey speckle field where the reference is air, because its reverse bridge
terminates before ``t=0`` and leaves a residual noise floor. Nothing in the run
noticed: the training loss was finite and falling, so the crash detectors were
quiet and the arm reported ``success: True``.

This module supplies the missing question — *does the prediction render air as
air?* — as two scale-invariant statistics measured against the reference's own
background:

``air_level``
    Mean prediction value where the **reference** is air. A healthy
    reconstruction leaves the air region dark; a non-converged reverse chain,
    a collapsed posterior, or an off-scale guidance step fills it.

``blank_excess``
    How much more background the prediction has than the reference. Catches the
    opposite degeneracy — a dead output that is blank where there is anatomy —
    which ``air_level`` alone reads as (spuriously) perfect.

Both are computed on per-sample min-max-normalised tensors, so they are
invariant to the output scale and directly comparable to the values measured
off the saved validation PNGs.

Thresholds are calibrated against the 2026-07 MRIxFields2026 cohort (68 arms,
8 visually confirmed degenerate). The separation is wide, not marginal:

===============================  ===========  ==========================
band                             air_level    arms
===============================  ===========  ==========================
healthy (60 arms)                <= 0.082     max ``b110_ilvr_anchor``
degenerate (7 of 8)              >= 0.245     min ``b35_field_cfg``
dead-black (1 of 8)              0.007        ``b110_ablate_no_anchor``
                                              (caught by ``blank_excess``)
===============================  ===========  ==========================

``AIR_LEVEL_LIMIT = 0.15`` sits in the middle of that 3x gap; on the calibration
cohort the pair of rules flags 8/8 confirmed-degenerate arms and 0/60 healthy
ones.

**Scope.** These rules resolve the *severe* class. They deliberately do not try
to separate the marginal band (``air_level`` 0.08-0.15), where a faint
background haze on an edge slice is indistinguishable from mild degeneracy
without looking at the picture. The measured values are always reported so a
human review can see that band; only the unambiguous cases are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

#: Mean prediction value over the reference's air region, above which the
#: output is judged to have filled the background. Midpoint of the measured
#: healthy/degenerate gap (0.082 .. 0.245).
AIR_LEVEL_LIMIT = 0.15

#: Excess background fraction (prediction minus reference) above which the
#: output is judged blank. Measured healthy maximum is +0.054
#: (``b35_ablate_guidance``); the dead-black ``b110_ablate_no_anchor`` is +0.242.
BLANK_EXCESS_LIMIT = 0.20

#: Fraction of the reference below which a pixel counts as air.
AIR_THRESHOLD = 0.05

#: Fraction of an image below which a pixel counts as background.
BACKGROUND_THRESHOLD = 0.10

#: Minimum reference air fraction for the guard to apply. Below this the image
#: is essentially all anatomy (a tight crop) and there is no background to grade.
#:
#: Must sit well above what normalisation alone manufactures: per-sample min-max
#: mapping sends the darkest pixels of ANY image to 0, so even an all-anatomy
#: crop reports a few percent of "air". Real MRI references in the calibration
#: cohort measure 0.72-0.78, so 0.15 never fires on genuine data while still
#: rejecting the manufactured case.
MIN_REFERENCE_AIR = 0.15

_EPS = 1e-8


@dataclass(frozen=True)
class OutputSanityReport:
    """Measured degeneracy statistics for one validation batch.

    ``verdict`` is ``None`` when the output is not flagged. It is a short
    machine-readable slug otherwise, and ``detail`` renders the numbers for a log
    line.
    """

    air_level: float
    blank_excess: float
    reference_air_fraction: float
    verdict: str | None
    detail: str

    @property
    def is_degenerate(self) -> bool:
        return self.verdict is not None


def _minmax_normalise(t: Tensor) -> tuple[Tensor, Tensor]:
    """Per-sample min-max normalise to ``[0, 1]``; also return the raw range.

    Matches what the validation PNG writer does, so the thresholds calibrated
    off the saved images transfer to the live tensors.
    """
    flat = t.flatten(1)
    lo = flat.amin(dim=1).view(-1, *([1] * (t.ndim - 1)))
    hi = flat.amax(dim=1).view(-1, *([1] * (t.ndim - 1)))
    rng = hi - lo
    return (t - lo) / rng.clamp_min(_EPS), rng


def measure_output_sanity(
    prediction: Tensor,
    target: Tensor,
    *,
    air_level_limit: float = AIR_LEVEL_LIMIT,
    blank_excess_limit: float = BLANK_EXCESS_LIMIT,
) -> OutputSanityReport:
    """Grade ``prediction`` against ``target`` for output degeneracy.

    Both tensors are ``[B, C, H, W]`` (or any shape whose first dim is the
    batch). Never raises on odd input: a reference with too little air, or a
    shape mismatch, yields a ``None`` verdict rather than aborting validation.
    """
    if prediction.shape != target.shape or prediction.numel() == 0:
        return OutputSanityReport(
            float("nan"), float("nan"), float("nan"), None, "shape mismatch or empty"
        )

    pred = prediction.detach().float()
    tgt = target.detach().float()
    pred_n, pred_rng = _minmax_normalise(pred)
    tgt_n, _ = _minmax_normalise(tgt)

    air = tgt_n < AIR_THRESHOLD
    ref_air_frac = float(air.float().mean())
    if ref_air_frac < MIN_REFERENCE_AIR:
        return OutputSanityReport(
            float("nan"),
            float("nan"),
            ref_air_frac,
            None,
            f"reference air fraction {ref_air_frac:.3f} below "
            f"{MIN_REFERENCE_AIR} — no background to grade",
        )

    air_level = float(pred_n[air].mean())
    pred_bg = float((pred_n < BACKGROUND_THRESHOLD).float().mean())
    tgt_bg = float((tgt_n < BACKGROUND_THRESHOLD).float().mean())
    blank_excess = pred_bg - tgt_bg

    detail = (
        f"air_level={air_level:.3f} (limit {air_level_limit}), "
        f"blank_excess={blank_excess:+.3f} (limit {blank_excess_limit}), "
        f"reference_air={ref_air_frac:.3f}"
    )

    # A constant output has no dynamic range at all — min-max normalisation of it
    # is meaningless, so check the raw range before trusting the ratios above.
    if bool((pred_rng < _EPS).all()):
        return OutputSanityReport(
            air_level,
            blank_excess,
            ref_air_frac,
            "constant_output",
            f"prediction is constant across every sample; {detail}",
        )
    if air_level > air_level_limit:
        return OutputSanityReport(
            air_level,
            blank_excess,
            ref_air_frac,
            "air_filled",
            f"prediction fills the reference's air region; {detail}",
        )
    if blank_excess > blank_excess_limit:
        return OutputSanityReport(
            air_level,
            blank_excess,
            ref_air_frac,
            "blank_output",
            f"prediction is blank where the reference has anatomy; {detail}",
        )
    return OutputSanityReport(air_level, blank_excess, ref_air_frac, None, detail)


__all__ = [
    "AIR_LEVEL_LIMIT",
    "BLANK_EXCESS_LIMIT",
    "OutputSanityReport",
    "measure_output_sanity",
]
