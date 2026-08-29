"""Config schema for the Fisher-Rao geodesic translation strategy (B-3.4)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class FisherRaoGeodesicConfig(StrictSchema):
    """Knobs for the Fisher-Rao geodesic cross-field translator (B-3.4).

    The information geometry (the Bernoulli-sphere geodesic step) is STRUCTURAL on the model
    (``model_kwargs.use_fisher_rao_geometry``); this block only carries the loss weight.
    """

    lambda_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the L1 reconstruction loss (geodesic-shot source vs target).",
    )
