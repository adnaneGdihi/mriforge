"""Checkpoint management configuration schema."""

from pydantic import BaseModel, Field


class CheckpointConfigSchema(BaseModel):
    """Checkpoint saving and loading configuration.

    Defines how and when model checkpoints are saved and loaded.

    Example:
        >>> config = CheckpointConfigSchema(
        ...     enabled=True,
        ...     save_interval=1000,
        ...     checkpoint_dir="./checkpoints",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
        "populate_by_name": True,
    }

    enabled: bool = Field(
        default=True,
        description="Enable checkpoint saving",
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory to save checkpoints",
        validation_alias="save_dir",
    )
    save_interval: int = Field(
        default=1000,
        ge=1,
        description="Save checkpoint every N steps",
    )
    save_on_epoch: bool = Field(
        default=True,
        description="Save checkpoint at end of each epoch",
    )
    keep_last_n: int = Field(
        default=5,
        ge=1,
        description="Keep only last N checkpoints",
    )
    keep_best_n: int = Field(
        default=3,
        ge=1,
        description="Keep best N checkpoints by metric",
    )
    pretrained_path: str | None = Field(
        default=None,
        description="Path to pretrained checkpoint to load",
    )
    load_optimizer_state: bool = Field(
        default=True,
        description="Load optimizer state from checkpoint",
    )
    load_scheduler_state: bool = Field(
        default=True,
        description="Load scheduler state from checkpoint",
    )
    resume_training: bool = Field(
        default=False,
        description="Resume training from checkpoint (load epoch/step counter)",
    )
    format: str = Field(
        default="safetensors",
        description="Format for model weight checkpoints (safetensors or pth)",
    )
    produced_by_arm: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Name of the upstream campaign arm that produces this arm's declared "
            "checkpoint(s). When set, check_checkpoint_existence treats a "
            "still-absent, campaign-artefact-rooted checkpoint (under "
            "experiments/active/ or experiments/results/) as a deferred campaign "
            "dependency — info-passing the standalone --strict pre-flight rather "
            "than hard-failing it — because the campaign runner builds the "
            "producer first. The real existence gate still fires at actual "
            "checkpoint-load time. It does NOT defer non-artefact paths or "
            "precomputed data artefacts (marker basis/template tensors)."
        ),
    )


__all__ = ["CheckpointConfigSchema"]
