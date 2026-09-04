"""Config for UlfMapStrategy (MICCAI MRIxFields2026, B-2.1 physics-noise MAP).

Read by ``UlfMapStrategy`` via ``config.training.ulf_map``. The ULF→HF restoration
is a MAP / half-quadratic-splitting solve whose data term is weighted by the
physically-calibrated Hoult noise level ``σ ∝ B0^{-7/4}`` (lower field ⇒ noisier
measurement ⇒ weaker data pull ⇒ the learned prior fills more). The denoiser /
ADMM iterations / penalty are inherited from ``config.training.pnp``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class UlfMapConfig(StrictSchema):
    """Ultra-low-field physics-noise MAP knobs (B-2.1)."""

    source_field: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Ultra-low-field measurement field strength (Tesla). Sets the Hoult "
            "noise level via σ = sigma_floor / (source_field/3)^1.75."
        ),
    )
    sigma_floor: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "High-field reference noise std (the 3T baseline) used to derive the "
            "ULF noise level through the Hoult B0^{7/4} scaling."
        ),
    )
    data_weight_max: float = Field(
        default=1.0e3,
        gt=0.0,
        description=(
            "Upper clamp on the data-consistency weight 1/σ² (numerical stability "
            "at high field where σ is tiny)."
        ),
    )
    blur_cutoff: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Low-pass cutoff (normalised frequency) of the field-dependent forward "
            "operator A modelling the ULF effective-resolution loss. Smaller ⇒ more "
            "blur; the data term constrains the passband and the prior fills the "
            "stopband."
        ),
    )
