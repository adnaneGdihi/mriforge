"""Config schema for the Fisher-Rao geodesic translation strategy (B-3.4)."""

from __future__ import annotations

from spectramr.config.schemas.strictness import StrictSchema


class FisherRaoGeodesicConfig(StrictSchema):
    """Knobs for the Fisher-Rao geodesic cross-field translator (B-3.4).

    The information geometry (the Bernoulli-sphere geodesic step) is STRUCTURAL on the model
    (``model_kwargs.use_fisher_rao_geometry``); this block only carries the loss weight.
    """
