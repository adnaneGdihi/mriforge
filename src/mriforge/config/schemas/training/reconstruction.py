"""Reconstruction-specific training configuration schema.

This module defines TrainingConfigReconstruction for image reconstruction paradigms.
It enforces reconstruction objectives and validates reconstruction-specific fields.
"""

from pydantic import Field, field_validator

from ..objectives import ObjectiveConfigSchema
from .base import BaseTrainingConfigSchema


class TrainingConfigReconstruction(BaseTrainingConfigSchema):
    """Image Reconstruction training configuration.

    Enforces reconstruction objectives and validates that:
    - Reconstruction objective configuration is present
    - Loss weights are valid
    - Data consistency settings are appropriate

    Example:
        >>> config = TrainingConfigReconstruction(
        ...     training_mode="reconstruction",
        ...     epochs=100,
        ...     objectives=ObjectiveConfigSchema(
        ...         reconstruction=ReconstructionObjectiveConfig(
        ...             lambda_l1=10.0,
        ...             lambda_perceptual=5.0
        ...         )
        ...     ),
        ... )
    """

    # Override with Reconstruction objectives
    objectives: ObjectiveConfigSchema = Field(
        default_factory=ObjectiveConfigSchema,
        description="Training objectives (Reconstruction must have objectives.reconstruction configured)",
    )

    # Reconstruction-specific features
    enable_data_consistency: bool = Field(
        default=False,
        description="Enable physics-based data consistency",
    )
    dc_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for data consistency loss",
    )
    dc_interval: int = Field(
        default=1,
        ge=1,
        description="Apply data consistency every N iterations",
    )
    enable_coil_sensitivity_estimation: bool = Field(
        default=False,
        description="Estimate coil sensitivities during training",
    )
    enable_kspace_recon: bool = Field(
        default=False,
        description="Reconstruct in k-space domain",
    )
    enable_iterative_refinement: bool = Field(
        default=False,
        description="Enable iterative refinement during inference",
    )
    num_refinement_steps: int = Field(
        default=1,
        ge=1,
        description="Number of refinement iterations",
    )

    # Reconstruction-specific regularization
    enable_tv_regularization: bool = Field(
        default=False,
        description="Enable total variation regularization",
    )
    tv_weight: float = Field(
        default=0.01,
        ge=0,
        description="Weight for TV regularization",
    )
    enable_wavelets: bool = Field(
        default=False,
        description="Use wavelet domain constraints",
    )
    wavelet_weight: float = Field(
        default=0.01,
        ge=0,
        description="Weight for wavelet constraints",
    )

    @field_validator("objectives")
    @classmethod
    def validate_recon_objectives(cls, obj: ObjectiveConfigSchema) -> ObjectiveConfigSchema:
        """Objectives accepted as-is; loss weights live under ``losses.reconstruction`` (v6.0+).

        The previous body had two defects: ``if obj is None: obj = ...`` was a dead
        local rebind (Pydantic passes the already-validated value, never None given
        the ``default_factory``), and ``if obj.reconstruction is None: raise`` fired
        on *every* default construction because ``ObjectiveConfigSchema.reconstruction``
        defaults to ``None`` in v6.0 — making a bare ``TrainingConfigReconstruction()``
        unconstructable. Loss weights migrated to ``losses.reconstruction``; this
        validator is now a pass-through.

        Args:
            obj: ObjectiveConfigSchema to validate

        Returns:
            The objectives schema unchanged.
        """
        return obj

    @field_validator("dc_interval")
    @classmethod
    def validate_dc_interval(cls, v: int) -> int:
        """Ensure data consistency interval is positive."""
        if v <= 0:
            raise ValueError("dc_interval must be > 0")
        return v

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


__all__ = ["TrainingConfigReconstruction"]
