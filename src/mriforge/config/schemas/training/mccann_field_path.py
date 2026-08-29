"""Config schema for the McCann single-potential field-path strategy (B-3.9)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class McCannFieldPathConfig(StrictSchema):
    """Knobs for the McCann geodesic field-path strategy (B-3.9).

    The convexity (Brenier guarantee) is STRUCTURAL on the model
    (``model_kwargs.enforce_convexity``); the path function is learnable on the model. This
    block only carries the loss weight.
    """

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (McCann-interpolated source vs target).",
    )
