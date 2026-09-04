"""Config for FieldGuidedDiffusionStrategy (MICCAI MRIxFields2026, B-3.5).

Read by ``FieldGuidedDiffusionStrategy`` at construction via
``config.training.field_guided_diffusion``; absence falls back to defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema


class FieldGuidedDiffusionConfig(StrictSchema):
    """Continuous-field classifier-free-guidance diffusion knobs (B-3.5)."""

    timesteps: int = Field(
        default=1000,
        ge=1,
        description="Number of DDPM diffusion steps (length of alphas_cumprod).",
    )
    beta_schedule: Literal["cosine", "linear"] = Field(
        default="cosine",
        description="Noise schedule for alphas_cumprod ('cosine' or 'linear').",
    )
    guidance_prob: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of dropping the continuous-field conditioning during "
            "training (the score net's learned null-field FiLM takes over). 0 trains "
            "no null path -> CFG at sampling is undefined; keep > 0 for field-CFG."
        ),
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        description=(
            "Classifier-free guidance weight at sampling: "
            "eps = eps_null + guidance_scale*(eps_field - eps_null). 1.0 = plain "
            "conditional sampling (no guidance) — the one-knob ablation control."
        ),
    )
    sampling_steps: int = Field(
        default=50,
        ge=1,
        description="DDIM reverse-sampling steps at validation (<= timesteps).",
    )

    @model_validator(mode="after")
    def _guidance_requires_dropout(self) -> FieldGuidedDiffusionConfig:
        """Reject the degenerate CFG combo at load time (pitfall #9/#15/#20).

        The classifier-free null path is trained ONLY via field dropout (prob
        ``guidance_prob``). With ``guidance_prob == 0`` the learned null FiLM never
        receives a gradient and stays at its identity init, so guided sampling
        (``guidance_scale != 1``) extrapolates against an UNTRAINED reference —
        scientifically meaningless yet crash-free. Require dropout whenever guidance
        is requested; ``guidance_scale == 1`` (plain conditional, the null forward is
        never invoked) remains legal with any ``guidance_prob``.
        """
        if self.guidance_prob == 0.0 and self.guidance_scale != 1.0:
            raise ValueError(
                "field_guided_diffusion: guidance_prob=0 trains no null path, so "
                f"classifier-free guidance is undefined for guidance_scale="
                f"{self.guidance_scale}. Set guidance_prob>0 for guidance (or "
                "guidance_scale=1 for plain conditional sampling)."
            )
        return self
