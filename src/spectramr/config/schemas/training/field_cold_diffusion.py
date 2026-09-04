"""Config for FieldColdDiffusionStrategy (MICCAI MRIxFields2026, B-2.4).

Cold diffusion (Bansal et al. 2022) with a *deterministic field down-conversion*
degradation instead of Gaussian noising: the degradation path runs from the
high-field image (t=0, near-identity) to a ULF-like low-resolution image (t=T, a
strong low-pass). A restorer is trained to invert the path and the image is
recovered by the improved cold-diffusion sampling rule. Read via
``config.training.field_cold_diffusion``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class FieldColdDiffusionConfig(StrictSchema):
    """Field-degradation cold-diffusion knobs (B-2.4)."""

    n_steps: int = Field(
        default=10,
        ge=1,
        description="Sampler step count (reverse cold-diffusion trajectory length).",
    )
    train_steps: int = Field(
        default=10,
        ge=1,
        description=(
            "Training curriculum resolution (t ~ U{1..train_steps}); INDEPENDENT of "
            "the sampler n_steps so the single-step ablation collapses only the "
            "sampler, not the training degradation distribution (pitfall #17)."
        ),
    )
    max_cutoff: float = Field(
        default=5.0,
        gt=0.0,
        description="Low-pass cutoff at t=0 (near-identity high-field end; large=sharp).",
    )
    min_cutoff: float = Field(
        default=0.25,
        gt=0.0,
        description="Low-pass cutoff at t=T (ULF end; small=heavy resolution loss).",
    )
