"""Config schema for the Brenier OT-map synthesis strategy (B-1.5)."""

from __future__ import annotations

from spectramr.config.schemas.strictness import StrictSchema


class BrenierSynthesisConfig(StrictSchema):
    """Knobs for the source-conditioned Brenier OT-map strategy (B-1.5).

    The convexity (Brenier guarantee) is STRUCTURAL on the model
    (``model_kwargs.enforce_convexity``); this block only carries the loss weight.
    """
