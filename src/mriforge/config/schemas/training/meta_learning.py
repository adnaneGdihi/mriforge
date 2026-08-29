"""Meta-learning training configuration schema.

This module defines the schema for meta-learning experiments, specifically
Model-Agnostic Meta-Learning (MAML) and its variants.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..data import DataConfigSchema
from ..loss import ReconstructionLossesConfig
from .base import BaseTrainingConfigSchema


class MetaLearningObjectivesConfig(BaseModel):
    """Objectives configuration for meta-learning."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    reconstruction: ReconstructionLossesConfig = Field(
        default_factory=ReconstructionLossesConfig,
        description="Reconstruction losses",
    )


class MetaLearningDataConfig(DataConfigSchema):
    """Data configuration for meta-learning."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",  # Allow extra fields
        "frozen": True,
    }

    dataset_type: str = Field(default="meta_learning")

    # Meta-learning specific fields
    task_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for task sampling (e.g., acceleration factors)",
    )
    support_size: int = Field(
        default=4,
        ge=1,
        description="Number of support examples per task",
    )
    query_size: int = Field(
        default=4,
        ge=1,
        description="Number of query examples per task",
    )
    tasks_per_epoch: int = Field(
        default=100,
        ge=1,
        description="Number of tasks to sample per epoch",
    )

    @field_validator("dataset_type")
    @classmethod
    def validate_dataset_type(cls, value: str) -> str:
        # Allow meta_learning and other types
        """validate_dataset_type.

        Args:
            value (str): Description.
        Returns:
            str: Description.
        """
        return value


class AdaptationConfig(BaseModel):
    """Configuration for inner-loop adaptation."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    adaptation_steps: int = Field(
        default=5,
        ge=1,
        description="Number of inner-loop gradient steps",
    )
    adaptation_lr: float = Field(
        default=0.01,
        ge=0.0,
        description="Learning rate for inner-loop adaptation",
    )
    meta_lr: float = Field(
        default=0.001,
        ge=0.0,
        description="Learning rate for outer-loop meta-update",
    )
    task_batch_size: int = Field(
        default=4,
        ge=1,
        description="Number of tasks sampled per meta-update",
    )
    adaptation_layers: str = Field(
        default="both",
        description="Which layers to adapt: 'all', 'heads', 'encoder', 'decoder', or 'both'",
    )
    use_memory_bank: bool = Field(
        default=False,
        description="Use memory bank for task embeddings",
    )
    memory_bank_size: int = Field(
        default=256,
        ge=1,
        description="Size of the memory bank",
    )


class MetaLearningModelConfig(BaseModel):
    """Model configuration for meta-learning."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",  # Allow extra fields for base model config
        "frozen": True,
    }

    model_type: Literal["meta_learning_friendly"] = "meta_learning_friendly"
    base_model_type: str = Field(
        default="standard_unet",
        description="Type of the base model to be meta-learned",
    )
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=1, ge=1)

    adaptation_config: AdaptationConfig = Field(
        default_factory=AdaptationConfig,
        description="Configuration for inner-loop adaptation",
    )

    @property
    def spatial_dims(self) -> int:
        """Required for shape validation. Meta-learning typically 2D."""
        return 2


class TrainingConfigMetaLearning(BaseTrainingConfigSchema):
    """Configuration for meta-learning training paradigm."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    training_mode: Literal["meta_learning"] = "meta_learning"

    model: MetaLearningModelConfig = Field(
        ...,
        description="Meta-learning model configuration",
    )

    objectives: MetaLearningObjectivesConfig = Field(
        default_factory=MetaLearningObjectivesConfig,
        description="Training objectives configuration",
    )

    data: MetaLearningDataConfig = Field(
        ...,
        description="Meta-learning data configuration",
    )
