"""Config schema for the cross-source confluence (Fréchet-mean consensus) strategy (B-1.1)."""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class ConfluenceConfig(StrictSchema):
    """Knobs for the multi-source Fréchet-mean consensus strategy (B-1.1).

    Requires ``data.mrixfields_pairing_policy='multi_source'`` (the travelling-volunteer
    tuple). ``lambda_consensus`` weights the Fréchet-mean regulariser; ``lambda_consensus=0``
    is the one-knob control (per-source fidelity only, no consensus — still a valid
    multi-source translator).
    """

    lambda_consensus: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the pairwise consensus (Fréchet-mean) regulariser. 0 = "
        "per-source fidelity only (the one-knob ablation).",
    )
    contrast_conditioning: bool = Field(
        default=True,
        description="Condition the renderer on the contrast one-hot (matches B-3.8/B-1.9).",
    )
