"""Config schema for the deep-VIB recoverability-ceiling strategy (B-2.3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class RecoverabilityVIBConfig(StrictSchema):
    """Knobs for the deep variational information-bottleneck restorer (B-2.3).

    ``beta`` is the IB Lagrange multiplier (the one-knob): higher ``beta`` tightens the rate
    budget ``I(X;Z)`` and lowers recoverable quality; the low-``beta`` limit is the recoverability
    ceiling. ``latent_channels`` (the bottleneck width) lives on ``model_kwargs``.
    """

    beta: float = Field(
        default=1.0e-3,
        ge=0.0,
        description="IB Lagrange multiplier on the rate KL(q(z|x)||N(0,I)); 0 => plain autoencoder.",
    )
    lambda_recon: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the reconstruction (restoration) loss.",
    )
    beta_warmup_iters: int = Field(
        default=0,
        ge=0,
        description=(
            "Linear beta warmup horizon (iterations). The effective multiplier ramps "
            "beta_eff = beta * min(1, iteration / beta_warmup_iters) from 0 to the target, so "
            "the reconstruction term establishes a faithful z-usage BEFORE the rate pressure "
            "tightens — the anti-posterior-collapse guard (#20) that keeps the tight-budget arm "
            "from collapsing to a measurement-independent DC blob. It is the 'recon floor': recon "
            "leads early. Identical in both arms (one-knob stays beta); only shapes the first "
            "beta_warmup_iters of the schedule. 0 disables (immediate full beta)."
        ),
    )
    rate_reduction: Literal["sum", "mean_per_dim"] = Field(
        default="sum",
        description=(
            "How the rate KL is reduced across the SPATIAL latent before beta scales it. "
            "'sum' (legacy default) sums every latent dimension and divides only by batch, "
            "so the rate — and therefore the effective beta — grows with the latent grid. "
            "On the b23 arm the latent is 16x91x109 = 158,704 dims, which makes the "
            "innocuous-looking beta=1e-3 equivalent to a PER-DIMENSION beta of 159 (and the "
            "tight arm's 1e-1 equivalent to 15,870) — three to five orders of magnitude past "
            "the collapse threshold. That is why the encoder collapsed again on 2026-07-23 "
            "despite beta_warmup_iters AND free_bits, and why BOTH beta ends previously "
            "landed at the same SSIM: the rate-distortion sweep has no live region. "
            "'mean_per_dim' divides by the latent dimension count, making beta dimensionless "
            "and portable across latent geometry (the usual VIB convention, where beta ~ "
            "1e-3..1e-1 is the useful range). Default stays 'sum' so no existing arm's "
            "objective changes silently; opt in per arm."
        ),
    )
    free_bits: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Free-bits floor on the rate PENALTY, in the units set by rate_reduction "
            "(nats/sample under 'sum', nats/dimension under 'mean_per_dim'): the loss "
            "penalises max(rate, free_bits), so KL(q(z|x)||N(0,I)) is never pushed below "
            "this floor (clamp has zero gradient below it). NOTE the units trap this "
            "guard fell into: under 'sum' a floor of 0.2 on the b23 arm is 1.3e-6 "
            "nats/dim, and the fully-collapsed encoder already sits at 0.044 aggregate — "
            "the floor was only 4.6x above total collapse and bought nothing. Free bits "
            "also only removes DOWNWARD pressure; it never pushes the rate back up, so it "
            "cannot rescue an encoder that has already collapsed (the 2026-07-23 repeat). "
            "It prevents drift, so it must be set in units that bind from the start — "
            "which is what 'mean_per_dim' makes possible. The REPORTED "
            "val_recoverability_rate stays the true "
            "unclamped KL, so the recoverability-budget measurement stays honest above the "
            "floor. Distinct from KL-capacity annealing (which PINS the rate to a target C and "
            "would corrupt the measurement — deliberately not used here). 0 disables."
        ),
    )
