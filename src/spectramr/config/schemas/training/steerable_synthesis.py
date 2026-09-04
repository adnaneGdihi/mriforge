"""Config schema for the steerable C_4-equivariant synthesis strategy (B-1.4)."""

from __future__ import annotations

from spectramr.config.schemas.strictness import StrictSchema


class SteerableSynthesisConfig(StrictSchema):
    """Knobs for the steerable equivariant cross-field synthesis strategy (B-1.4).

    The equivariance itself is STRUCTURAL on the model (``model_kwargs.use_equivariance``);
    this block only carries the loss weight. The render is an L1 reconstruction of the
    target from the source, conditioned on the source field.
    """
