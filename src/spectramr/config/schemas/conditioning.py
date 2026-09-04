"""Adaptive-conditioning schema (2026-05-26).

The YAML-facing knob that lets an experiment declare WHICH condition sources
the model should adapt to — degradation severity, diffusion timestep,
acquisition order/params, contrast, or motion pose. Consumed by
``spectramr.models.conditioning.AdaptiveConditioner``.

Layer note: this module lives in the innermost ``config`` layer and therefore
MUST NOT import from ``models``. The Literal source set below is mirrored from
the conditioner registry; a test
(``tests/unit/config/test_conditioning_schema.py``) asserts the two stay in
sync, so drift is caught without an upward import.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, model_validator

# Must equal the keys registered via ``@register_conditioner`` in
# ``spectramr/models/conditioning/encoders.py`` + ``koopman_motion.py``.
ConditioningSource = Literal[
    "diffusion_t",
    "severity_vec",
    "acq_time_tau",
    "acquisition",
    "contrast_id",
    "motion_pose",
    "motion_koopman",
    # Domain axes (2026-05-26): paired field-strength (ULF↔HF), multi-scanner /
    # vendor harmonisation, and multi-site / federated conditioning. See
    # docs/conditioning_embedding_plan.rst.
    "field_strength",
    "scanner_id",
    "site_id",
]


class ConditioningConfig(BaseModel):
    """Declarative configuration for adaptive conditioning.

    Example:
        >>> ConditioningConfig(
        ...     enabled=True,
        ...     sources=["diffusion_t", "severity_vec"],
        ...     fusion="concat",
        ... )
    """

    model_config = {"extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description="Enable adaptive conditioning (FiLM from the active sources).",
    )
    sources: list[ConditioningSource] = Field(
        default_factory=list,
        description="Condition sources to encode and fuse. Each must be a "
        "registered conditioner key.",
    )
    embed_dim: int = Field(
        default=128,
        gt=0,
        description="Per-source embedding width AND fused conditioning width.",
    )
    fusion: Literal["concat", "sum"] = Field(
        default="concat",
        description="How per-source embeddings are fused: concat→linear, or sum.",
    )
    injection: Literal["film"] = Field(
        default="film",
        description="How the fused vector modulates features. Only FiLM for now.",
    )
    encoder_kwargs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-source encoder constructor overrides, e.g. "
        '{"severity_vec": {"num_severities": 6}}.',
    )

    @model_validator(mode="after")
    def _enabled_needs_sources(self) -> ConditioningConfig:
        if self.enabled and not self.sources:
            raise ValueError("conditioning.enabled=True requires at least one source.")
        return self

    @classmethod
    def allowed_sources(cls) -> tuple[str, ...]:
        """The set of legal source strings (mirrors the conditioner registry)."""
        return get_args(ConditioningSource)


__all__ = ["ConditioningConfig", "ConditioningSource"]
