"""VAE/VQVAE-specific training configuration schema.

This module defines TrainingConfigVAE for variational autoencoder training.
It enforces latent objectives and validates VAE-specific fields.
"""

from pydantic import Field, ValidationInfo, field_validator

from ..objectives import ObjectiveConfigSchema
from .base import BaseTrainingConfigSchema


class TrainingConfigVAE(BaseTrainingConfigSchema):
    """Variational Autoencoder training configuration.

    Enforces latent objectives and validates that:
    - Latent objective configuration is present
    - KL weight is reasonable
    - Reconstruction weight is valid

    Example:
        >>> config = TrainingConfigVAE(
        ...     training_mode="vae",
        ...     epochs=100,
        ...     objectives=ObjectiveConfigSchema(
        ...         latent=LatentObjectiveConfig(
        ...             latent_dim=128,
        ...             beta_kl=0.01
        ...         )
        ...     ),
        ... )
    """

    # Override with VAE objectives
    objectives: ObjectiveConfigSchema = Field(
        default_factory=ObjectiveConfigSchema,
        description="Training objectives (VAE must have objectives.latent configured)",
    )

    # VAE-specific features
    enable_vq_commitment: bool = Field(
        default=True,
        description="Enable VQ-VAE commitment loss",
    )
    enable_kl_annealing: bool = Field(
        default=False,
        description="Enable KL weight annealing during training",
    )
    kl_anneal_steps: int = Field(
        default=10000,
        ge=0,
        description="Number of steps over which to anneal KL weight",
    )
    kl_anneal_start: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Starting KL weight for annealing",
    )
    kl_anneal_end: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Ending KL weight for annealing",
    )
    enable_temperature_annealing: bool = Field(
        default=False,
        description="Enable temperature annealing for gumbel-softmax",
    )
    temperature_start: float = Field(
        default=1.0,
        gt=0,
        description="Starting temperature",
    )
    temperature_end: float = Field(
        default=0.5,
        gt=0,
        description="Ending temperature",
    )
    temperature_anneal_steps: int = Field(
        default=10000,
        ge=0,
        description="Steps for temperature annealing",
    )
    perplexity_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for perplexity metric tracking",
    )

    @field_validator("objectives")
    @classmethod
    def validate_vae_objectives(cls, obj: ObjectiveConfigSchema) -> ObjectiveConfigSchema:
        """Ensure VAE/latent objectives are configured.

        Args:
            obj: ObjectiveConfigSchema to validate

        Returns:
            Validated ObjectiveConfigSchema

        Raises:
            ValueError: If latent objectives are missing or invalid
        """
        if obj.latent is None or obj.latent.latent_dim is None:
            raise ValueError(
                "VAE training requires objectives.latent configuration with latent_dim"
            )
        return obj

    @field_validator("kl_anneal_end")
    @classmethod
    def validate_kl_anneal_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure kl_anneal_end >= kl_anneal_start."""
        if "kl_anneal_start" in info.data and v < info.data["kl_anneal_start"]:
            raise ValueError("kl_anneal_end must be >= kl_anneal_start")
        return v

    @field_validator("temperature_end")
    @classmethod
    def validate_temperature_range(cls, v: float, info: ValidationInfo) -> float:
        """Ensure temperature_end > 0."""
        if v <= 0:
            raise ValueError("temperature_end must be > 0")
        return v

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


__all__ = ["TrainingConfigVAE"]
