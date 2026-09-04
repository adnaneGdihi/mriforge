"""Config for MonotoneFieldStrategy (MICCAI MRIxFields2026, B-2.6).

Read by ``MonotoneFieldStrategy`` via ``config.training.monotone_field``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class MonotoneFieldConfig(StrictSchema):
    """Monotone field-ordered generator knobs (B-2.6).

    The structural monotonicity knob ``enforce_monotone`` lives on the MODEL
    (``model.model_kwargs.enforce_monotone``), since the model needs it to choose
    softplus vs raw coefficients; the one-knob ablation flips it there.
    """

    lambda_monotone: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Optional soft monotonicity penalty mean(relu(out(tau)-out(tau+d))) added "
            "on top of the structural guarantee (0 disables; the softplus head already "
            "guarantees monotonicity, so this is a belt-and-suspenders regulariser)."
        ),
    )
