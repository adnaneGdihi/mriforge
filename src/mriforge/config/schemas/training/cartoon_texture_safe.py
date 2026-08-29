"""Config schema for the cartoon-texture safe-restoration strategy (B-2.5)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class CartoonTextureSafeConfig(StrictSchema):
    """Knobs for the cartoon-texture (BV/G) hallucination-confinement prior (B-2.5).

    The decomposition is parameter-free; ``lambda_ct=0`` is the EXACT param-matched L1-only
    control. ``cartoon_weight >> texture_weight`` is the confinement bias (structure matched hard,
    texture loose).
    """

    lambda_l1: float = Field(
        default=1.0, ge=0.0, description="Weight of the L1 loss to the clean target."
    )
    lambda_ct: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of the cartoon-texture prior (0 => L1-only control).",
    )
    tv_weight: float = Field(
        default=0.15, ge=0.0, description="ROF TV weight for the cartoon extraction."
    )
    n_iter: int = Field(
        default=40, ge=1, description="ROF gradient-descent iterations (cartoon denoise)."
    )
    cartoon_weight: float = Field(
        default=1.0, ge=0.0, description="Weight of the HARD cartoon (structure) match."
    )
    texture_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight of the LOOSE texture (coarse-energy) match — penalises texture ABSENCE "
        "(over-smoothing) while staying cheap for same-energy texture swaps.",
    )
    texture_pool: int = Field(
        default=4, ge=1, description="Pool size for the coarse texture-energy match (larger=finer)."
    )
