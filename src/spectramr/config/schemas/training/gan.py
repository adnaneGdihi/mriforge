"""GAN-specific training configuration schema.

This module defines TrainingConfigGAN for adversarial training paradigms.
It enforces GAN objectives and validates GAN-specific fields.
"""

from pydantic import Field, ValidationInfo, field_validator

from ..objectives import ObjectiveConfigSchema
from .base import BaseTrainingConfigSchema


class TrainingConfigGAN(BaseTrainingConfigSchema):
    """GAN training configuration.

    Enforces GAN objectives and validates that:
    - GAN objective configuration is present
    - Discriminator learning rate is specified
    - GAN-specific fields are valid

    Example:
        >>> config = TrainingConfigGAN(
        ...     training_mode="gan",
        ...     epochs=100,
        ...     discriminator_learning_rate=2.0e-4,
        ...     objectives=ObjectiveConfigSchema(
        ...         gan=GANObjectiveConfig(lambda_adv=1.0, lambda_gp=10.0)
        ...     ),
        ... )
    """

    # Override with GAN objectives
    objectives: ObjectiveConfigSchema = Field(
        default_factory=ObjectiveConfigSchema,
        description="Training objectives (GAN must have objectives.gan configured)",
    )

    # GAN-specific hyperparameters
    discriminator_learning_rate: float | None = Field(
        default=None,
        gt=0,
        description="Separate learning rate for discriminator (if None, uses learning_rate)",
    )
    disc_updates_per_gen: int = Field(
        default=1,
        ge=1,
        description="Number of discriminator updates per generator update",
    )

    # GAN-specific features
    enable_spectral_norm: bool = Field(
        default=False,
        description="Enable spectral normalization on discriminator weights",
    )
    enable_gradient_penalty: bool = Field(
        default=True,
        description="Enable gradient penalty regularization",
    )
    gradient_penalty_weight: float = Field(
        default=10.0,
        ge=0,
        description="Weight for gradient penalty loss",
    )
    enable_feature_matching: bool = Field(
        default=False,
        description="Enable feature matching loss between real and generated images",
    )
    feature_matching_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for feature matching loss",
    )

    @field_validator("objectives")
    @classmethod
    def validate_gan_objectives(cls, obj: ObjectiveConfigSchema) -> ObjectiveConfigSchema:
        """Ensure GAN objectives are configured.

        Args:
            obj: ObjectiveConfigSchema to validate

        Returns:
            Validated ObjectiveConfigSchema

        Raises:
            ValueError: If GAN objectives are missing or invalid
        """
        if obj.gan is None or obj.gan.lambda_adv is None:
            raise ValueError("GAN training requires objectives.gan configuration with lambda_adv")
        return obj

    @field_validator("discriminator_learning_rate", mode="before")
    @classmethod
    def default_disc_lr(cls, v: float | None, info: ValidationInfo) -> float | None:
        """If unset, the strategy reads ``optimization.optimizer.learning_rate`` at runtime.

        The previous body tried ``info.data["learning_rate"]``, but there is no flat
        top-level ``learning_rate`` field on ``BaseTrainingConfigSchema`` (the
        canonical knob is nested ``optimization.optimizer.learning_rate``), so the fallback
        was permanently dead — ``discriminator_learning_rate`` stayed ``None``
        regardless. Kept as an explicit pass-through to document that semantics.
        """
        return v

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


__all__ = ["TrainingConfigGAN"]
