"""Config schema for the Doob h-transform 7T diffusion bridge strategy (B-1.10)."""

from __future__ import annotations

from pydantic import Field

from mriforge.config.schemas.strictness import StrictSchema


class DoobBridgeConfig(StrictSchema):
    """Knobs for the 7T-marginal-score SDEdit bridge (B-1.10).

    The marginal score net is trained by denoising score matching on the 7T target. Sampling
    is a score-based SDEdit reverse from a noised source. ``strength`` is the meaningful
    one-knob (source-anchored vs unconditional); ``h_scale`` scales the score (0 = a
    denoiser-off sanity floor, NOT a clean no-translation control); ``eta`` adds DDIM
    stochasticity; ``val_seed`` fixes the validation init for reproducible monitoring.
    """

    timesteps: int = Field(default=1000, ge=1, description="DDPM training timesteps.")
    beta_schedule: str = Field(
        default="cosine", description="Noise schedule ('cosine' or 'linear')."
    )
    sampling_steps: int = Field(
        default=50,
        ge=2,
        description="Reverse (DDIM) steps from the SDEdit init down to t=0 (>=2 so the "
        "final step reaches t=0; 1 would return a one-shot x0-estimate at t0).",
    )
    strength: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="SDEdit init level: noise the source to t0=strength*(T-1). <1 anchors to "
        "the source (preserves identity); 1 ~ pure noise -> unconditional 7T generation.",
    )
    h_scale: float = Field(
        default=1.0,
        ge=0.0,
        description="Scale on the 7T-marginal score. 1 = full score-based reverse "
        "(translation); 0 = zeroes the eps-net (denoiser-off sanity FLOOR, saturated noise).",
    )
    eta: float = Field(
        default=0.0,
        ge=0.0,
        description="DDIM stochasticity (the g dW term); 0 = deterministic reverse.",
    )
    val_seed: int = Field(
        default=0,
        ge=0,
        description="Fixed seed for the validation SDEdit init (reproducible monitoring).",
    )
    anchor_scale: int = Field(
        default=0,
        ge=0,
        le=64,
        description="ILVR-style source structure anchor (Choi et al. 2021): low-pass "
        "downsample factor N. Each reverse step replaces the low-frequency band (downsample "
        "by N, upsample back) of the iterate with the source's, pinning subject anatomy while "
        "the 7T score supplies high-frequency contrast/detail. 0/1 = OFF (bare bridge); >=2 = "
        "ON (larger N anchors only coarser structure, leaving more freedom to translate).",
    )
    residual: bool = Field(
        default=False,
        description="Predict the source->7T RESIDUAL (target-source), source-conditioned, and "
        "compose output=source+residual. Requires model_type doob_residual_score_unet (the net "
        "sees the source as an extra input channel). The paired/registered/normalized volumes "
        "make the residual small and high-frequency, concentrating capacity on subject-faithful "
        "detail the unconditional bridge misses. False = the unconditional 7T-marginal bridge. "
        "With anchor_scale>=2 the ILVR anchor is applied in residual space (the residual's "
        "low-frequency band is zeroed) so the output low-frequency equals the source exactly.",
    )
