"""Config for FieldConditionedINRStrategy (MICCAI MRIxFields2026, B-2.8).

Read by ``FieldConditionedINRStrategy`` via ``config.training.field_conditioned_inr``.
"""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class FieldConditionedINRConfig(StrictSchema):
    """Field-conditioned INR super-resolution knobs (B-2.8)."""

    use_field_conditioning: bool = Field(
        default=True,
        description=(
            "Condition the INR style/FiLM on the continuous target field. When False, "
            "a constant (zero) field is passed so the INR is field-agnostic — the "
            "one-knob ablation isolating the value of field conditioning."
        ),
    )
    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (rendered vs target).",
    )
