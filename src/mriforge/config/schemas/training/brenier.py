"""Config schema for the Brenier OT-map synthesis strategy (B-1.5)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class BrenierSynthesisConfig(StrictSchema):
    """Knobs for the source-conditioned Brenier OT-map strategy (B-1.5).

    The convexity (Brenier guarantee) is STRUCTURAL on the model
    (``model_kwargs.enforce_convexity``); this block only carries the loss weight.
    """

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (Brenier-mapped source vs 7T target).",
    )
