"""Config schema for the continuous low-rank modulation strategy (B-3.6)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class LoRAModulationConfig(StrictSchema):
    """Knobs for the continuous low-rank modulation translator (B-3.6).

    The low-rank weight modulation is STRUCTURAL on the model
    (``model_kwargs.lora_rank`` — the compliance rank bound); this block only carries the loss
    weight.
    """

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (modulated backbone vs target).",
    )
