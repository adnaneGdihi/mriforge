"""Test-Time Optimization (TTO) configuration schema.

This module defines the schema for TTO experiments (Stage 2 of motion correction).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .base import BaseTrainingConfigSchema


class TTOConfig(BaseModel):
    """Test-Time Optimization configuration."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    lambda_tv: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight for Total Variation (TV) regularization during TTO",
    )
    lambda_dc: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight for Data Consistency (DC) loss during TTO (must be > 0)",
    )
    max_translation: float = Field(
        default=5.0,
        gt=0.0,
        le=128.0,
        description="Maximum expected translation (pixels) for optimization bounds",
    )
    max_rotation: float = Field(
        default=0.05,
        gt=0.0,
        le=3.14159,
        description="Maximum expected rotation (radians) for optimization bounds",
    )
    learning_rate: float = Field(
        default=1.0e-3,
        gt=0.0,
        description="TTO inner-loop learning rate (must be > 0)",
    )
    num_steps: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Number of TTO optimization steps per sample",
    )

    @model_validator(mode="after")
    def _at_least_one_active(self) -> "TTOConfig":
        if self.lambda_tv == 0.0 and self.lambda_dc == 0.0:
            raise ValueError(
                "At least one of lambda_tv or lambda_dc must be > 0; "
                "otherwise TTO has no objective."
            )
        return self


class TrainingConfigTTO(BaseTrainingConfigSchema):
    """Configuration for Test-Time Optimization paradigm."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    training_mode: Literal["test_time_optimization", "tto"] = Field(
        default="test_time_optimization"
    )

    tto: TTOConfig = Field(
        default_factory=TTOConfig,
        description="TTO specific configuration",
    )


__all__ = ["TTOConfig", "TrainingConfigTTO"]
