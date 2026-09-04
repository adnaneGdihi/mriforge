"""Objectives configuration schema.

This module contains objective and loss configuration for different training paradigms:
GAN, reconstruction, diffusion, and latent models.
"""

from pydantic import BaseModel, Field

from .base import (
    BiophysicalFlowObjectiveConfig,
    DiffusionObjectiveConfig,
    DomainAdaptationObjectiveConfig,
    GANObjectiveConfig,
    LatentObjectiveConfig,
    ReconstructionObjectiveConfig,
    SSLObjectiveConfig,
)

__all__ = ["ObjectiveConfigSchema"]


class ObjectiveConfigSchema(BaseModel):
    """Training objectives and loss configuration.

    Organized by training paradigm with nested configurations.
    Access via config.objectives.gan.*, config.objectives.reconstruction.*, etc.

    Note: GAN and diffusion objectives are Optional with None default.
    This prevents spurious warnings about paradigm-specific fields being present
    in other training modes (e.g., VAE mode shouldn't warn about GAN fields).
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",  # Strict schema: objectives should not carry loss weights
        "frozen": True,
    }

    # Paradigm-specific objectives: Optional with None default to prevent cross-paradigm warnings
    gan: GANObjectiveConfig | None = Field(default=None)
    diffusion: DiffusionObjectiveConfig | None = Field(default=None)

    # Common objectives - now Optional and deprecated in v6.0
    reconstruction: ReconstructionObjectiveConfig | None = Field(
        default=None, description="[DEPRECATED] Use 'losses.reconstruction' instead."
    )
    latent: LatentObjectiveConfig | None = Field(
        default=None, description="[DEPRECATED] Use 'losses.latent' instead."
    )
    domain_adaptation: DomainAdaptationObjectiveConfig | None = Field(
        default=None, description="[DEPRECATED] Use 'losses.adaptation' instead."
    )
    biophysical_flow: BiophysicalFlowObjectiveConfig | None = Field(
        default=None, description="[DEPRECATED] Use 'losses.flow' instead."
    )
    ssl: SSLObjectiveConfig | None = Field(
        default=None, description="[DEPRECATED] Use 'losses.ssl' instead."
    )
