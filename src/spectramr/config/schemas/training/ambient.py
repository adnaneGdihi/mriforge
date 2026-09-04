"""Ambient Diffusion training sub-schema — Phase B step 5.

Knobs for
:class:`spectramr.infrastructure.training.strategies.ambient_diffusion_strategy.AmbientDiffusionStrategy`
(Daras et al., NeurIPS 2023; Aali et al., MRI A-DPS). The SSDU Λ/Θ split is
lifted onto a diffusion prior; ``theta_fraction`` controls the held-out density,
``ambient_weight`` weights the held-out k-space consistency against the diffusion
denoising term. Frozen, ``extra="forbid"``, mounted on
``TrainingStrategyConfigSchema``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AmbientTrainingConfigSchema(BaseModel):
    """Parameters for Ambient Diffusion (SSDU split lifted onto a diffusion prior)."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    theta_fraction: float = Field(
        default=0.4,
        gt=0.0,
        lt=1.0,
        description="Fraction of acquired k-space held out into Theta (ambient consistency set).",
    )
    ambient_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on the held-out k-space consistency term relative to the "
            "diffusion denoising term. The ambient reweighting hook."
        ),
    )
    denoise_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on the standard diffusion eps-denoising term.",
    )
    split_seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed for reproducible Lambda/Theta splits.",
    )


__all__ = ["AmbientTrainingConfigSchema"]
