"""Config for BlochFieldStrategy (MICCAI MRIxFields2026, B-1.8).

Read by ``BlochFieldStrategy`` via ``config.training.bloch_field``. The structural
``use_field_dispersion`` knob (and the acquisition/dispersion constants) live on the
MODEL (``model.model_kwargs``); the one-knob ablation flips it there.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from mriforge.config.schemas.strictness import StrictSchema


class BlochFieldConfig(StrictSchema):
    """Bloch quantitative-parameter-bottleneck knobs (B-1.8)."""

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (SPGR render vs target).",
    )


class BlochSynthConfig(StrictSchema):
    """Cross-field relaxometry inversion + Bloch resynthesis knobs (idea 2.1, Task 1).

    Read by ``BlochSynthesisStrategy`` via ``config.training.bloch_synth``. Acquisition
    constants (TR/TE/flip), the reference field and ``learn_beta_per_tissue`` /
    ``use_opaque_residual`` live on the MODEL (``model.model_kwargs`` of
    ``relaxometry_encoder``); this block carries the strategy-side loss weights, the
    source-contrast set (identifiability), and the differentiable segmenter backend.
    """

    source_contrasts: list[Literal["T1w", "T2w", "FLAIR"]] = Field(
        default=["T1w", "T2w", "FLAIR"],
        description="Source contrasts stacked as the encoder input. len >= 3 makes the "
        "pointwise relaxometry over-determined (Proposition 3); the "
        "bloch_synth_source_contrast_count Tier-1 check enforces it and that "
        "model.in_channels matches.",
    )
    target_field_tesla: float = Field(
        default=7.0, gt=0.0, le=7.0, description="Synthesis target field (<= 7 T)."
    )
    signal_model: Literal["spgr"] = Field(
        default="spgr", description="Frozen differentiable signal equation (SPGR)."
    )
    dispersion_beta_bounds: tuple[float, float] = Field(
        default=(0.3, 0.4),
        description="Physiological envelope for the dispersion exponent; the "
        "dispersion_prior loss penalises excursions.",
    )
    learn_beta_per_tissue: bool = Field(
        default=True,
        description="Mirror of the model knob (documented for provenance); the ablation "
        "sets model.model_kwargs.learn_beta_per_tissue=false with dispersion_beta=0.",
    )
    opaque_band: Literal["highpass"] = Field(
        default="highpass",
        description="Opaque-band projector for the learned residual (high-pass).",
    )
    segmenter_backend: Literal["none", "label_dice"] = Field(
        default="label_dice",
        description="Differentiable teacher for seg-consistency. 'label_dice' = local "
        "intensity-soft segmenter (runs anywhere); 'none' disables it (seg weight must "
        "be 0). Real SynthSeg is a cluster follow-up.",
    )
    source_consistency_weight: float = Field(default=1.0, ge=0.0)
    seg_consistency_weight: float = Field(default=0.5, ge=0.0)
    dispersion_prior_weight: float = Field(default=0.1, ge=0.0)
    residual_weight: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> BlochSynthConfig:
        lo, hi = self.dispersion_beta_bounds
        if not (lo < hi):
            raise ValueError(
                f"dispersion_beta_bounds must be (lo, hi) with lo < hi; got ({lo}, {hi})."
            )
        if self.segmenter_backend == "none" and self.seg_consistency_weight > 0:
            raise ValueError(
                "segmenter_backend='none' requires seg_consistency_weight=0 (else the "
                "seg-consistency term is advertised but never computed, pitfall #16)."
            )
        return self
