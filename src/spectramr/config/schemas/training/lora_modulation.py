"""Config schema for the continuous low-rank modulation strategy (B-3.6)."""

from __future__ import annotations

from spectramr.config.schemas.strictness import StrictSchema


class LoRAModulationConfig(StrictSchema):
    """Knobs for the continuous low-rank modulation translator (B-3.6).

    The low-rank weight modulation is STRUCTURAL on the model
    (``model_kwargs.lora_rank`` — the compliance rank bound); this block only carries the loss
    weight.
    """
