"""Config schema for the scattering-Besov detail-prior strategy (B-1.3)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class ScatteringBesovConfig(StrictSchema):
    """Knobs for the scattering-Besov detail prior (B-1.3).

    The detail prior (wavelet scattering + Besov regularity) is a fixed, parameter-free loss;
    these are its weights and filterbank shape. ``lambda_scatter=0`` is the EXACT param-matched
    L1-only ablation control.
    """

    lambda_l1: float = Field(
        default=1.0, ge=0.0, description="Weight of the L1 loss to the 7T target."
    )
    lambda_scatter: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of the scattering-Besov detail prior (0 => L1-only control).",
    )
    n_scales: int = Field(
        default=3, ge=1, description="Number of wavelet scales in the filterbank."
    )
    n_orientations: int = Field(
        default=4, ge=1, description="Number of orientations per scale in the filterbank."
    )
    besov_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Besov regularity exponent s; the 2^(j*s) weighting up-weights finer scales.",
    )
    scatter_weight: float = Field(
        default=1.0, ge=0.0, description="Internal weight of the scattering (modulus) component."
    )
    besov_weight: float = Field(
        default=1.0, ge=0.0, description="Internal weight of the Besov (regularity) component."
    )
