"""Self-Supervised Learning (SSL) specific training configuration schema.

This module defines TrainingConfigSSL for self-supervised learning paradigms.
It enforces SSL-specific objectives and validates SSL-specific fields.
"""

from pydantic import Field, ValidationInfo, field_validator

from ..base import SSLConfigSchema
from ..objectives import ObjectiveConfigSchema
from .base import BaseTrainingConfigSchema


class TrainingConfigSSL(BaseTrainingConfigSchema):
    """Self-Supervised Learning training configuration.

    Supports SSL paradigms like:
    - BYOL (Bootstrap Your Own Latent)
    - SimCLR
    - MoCo (Momentum Contrast)
    - DINO

    Example:
        >>> config = TrainingConfigSSL(
        ...     training_mode="ssl",
        ...     epochs=100,
        ...     ssl=SSLConfigSchema(
        ...         method="byol",
        ...         ema_decay=0.999
        ...     ),
        ... )
    """

    # Override with SSL objectives
    objectives: ObjectiveConfigSchema = Field(
        default_factory=ObjectiveConfigSchema,
        description="Training objectives (SSL uses reconstruction objectives as base)",
    )

    # SSL-specific configuration
    ssl: SSLConfigSchema = Field(
        default_factory=SSLConfigSchema,
        description="SSL-specific settings (method, momentum, temperature)",
    )

    # SSL-specific hyperparameters
    enable_momentum_encoder: bool = Field(
        default=True,
        description="Use momentum/EMA encoder (required for BYOL, MoCo)",
    )
    momentum_decay: float = Field(
        default=0.999,
        ge=0,
        le=1,
        description="Momentum coefficient for encoder updates",
    )
    enable_projection_head: bool = Field(
        default=True,
        description="Use projection head after backbone",
    )
    projection_dim: int | None = Field(
        default=256,
        ge=1,
        description="Dimension of projection head",
    )
    enable_predictor_head: bool = Field(
        default=True,
        description="Use predictor head (BYOL-specific)",
    )
    predictor_dim: int | None = Field(
        default=4096,
        ge=1,
        description="Dimension of predictor head",
    )

    # SSL-specific losses
    enable_cosine_similarity_loss: bool = Field(
        default=True,
        description="Use cosine similarity loss (standard for SSL)",
    )
    temperature: float = Field(
        default=0.07,
        gt=0,
        description="Temperature for contrastive loss (SimCLR, MoCo)",
    )
    enable_negative_sampling: bool = Field(
        default=False,
        description="Use negative sampling (for contrastive methods)",
    )
    num_negative_samples: int = Field(
        default=256,
        ge=1,
        description="Number of negative samples (MoCo queue size)",
    )

    # SSL-specific augmentation
    enable_strong_augmentation: bool = Field(
        default=True,
        description="Use strong augmentation pipeline",
    )
    augmentation_probability: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Probability of applying augmentation",
    )

    @field_validator("objectives")
    @classmethod
    def validate_ssl_objectives(cls, obj: ObjectiveConfigSchema) -> ObjectiveConfigSchema:
        """Objectives accepted as-is; SSL loss weights live under ``losses.ssl`` (v6.0+).

        The previous body dereferenced ``obj.reconstruction.lambda_l1``, but in v6.0
        ``ObjectiveConfigSchema.reconstruction`` defaults to ``None`` and
        ``ReconstructionObjectiveConfig`` no longer carries a ``lambda_l1`` field —
        so the check raised ``AttributeError`` on every default construction. Loss
        weights migrated to ``losses.*``; this validator is now a pass-through.

        Args:
            obj: ObjectiveConfigSchema to validate

        Returns:
            The objectives schema unchanged.
        """
        return obj

    @field_validator("momentum_decay")
    @classmethod
    def validate_momentum(cls, v: float, info: ValidationInfo) -> float:
        """Ensure momentum decay is valid."""
        if not (0 <= v <= 1):
            raise ValueError("momentum_decay must be in [0, 1]")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float, info: ValidationInfo) -> float:
        """Ensure temperature is positive."""
        if v <= 0:
            raise ValueError("temperature must be > 0")
        return v

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


__all__ = ["TrainingConfigSSL"]
