"""Config for FieldFNOStrategy (MICCAI MRIxFields2026, B-1.6/3.7).

Read by ``FieldFNOStrategy`` via ``config.training.field_fno``. The structural
``field_token_enable`` knob lives on the MODEL (``model.model_kwargs.field_token_enable``);
the one-knob ablation flips it there.
"""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class FieldFNOConfig(StrictSchema):
    """Field-conditioned FNO knobs (B-1.6/3.7)."""

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (rendered vs target).",
    )
