"""Config for CrossFieldTranslationStrategy (MICCAI MRIxFields2026, B-3.8 / B-1.9).

Read by ``CrossFieldTranslationStrategy`` at construction time via
``config.training.cross_field``; absence falls back to defaults (the strategy is
usable with a bare ``training_mode: cross_field_translation``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from mriforge.config.schemas.strictness import StrictSchema


class FieldCocycleConfig(StrictSchema):
    """Cocycle-consistent unified cross-field operator knobs (idea 4.2, Task 3).

    Read by ``FieldCocycleTranslationStrategy`` via ``config.training.field_cocycle``.
    The strategy computes paired-fidelity (L1), latent-cycle, cocycle, field-identity
    and adversarial (hinge) terms inline; each weight below is the STATIC default that
    a ``loss_schedule:`` curriculum rule (``target: cocycle_consistency`` /
    ``field_identity`` / ``adversarial``) can override per step. Identity+cocycle alone
    admit the trivial ``G=Id``; the ``field_cocycle_fidelity_nonzero`` Tier-1 guard
    requires ``adversarial_weight > 0`` whenever ``cocycle+identity > 0``.
    """

    reference_field_tesla: float = Field(
        default=3.0,
        gt=0.0,
        description="Reference field s0 (Theorem 6 trivialising section); the "
        "canonicaliser Phi maps every field to this one.",
    )
    field_min_tesla: float = Field(
        default=0.1,
        gt=0.0,
        description="Lower end of the log-field axis (Tesla); used to normalise the "
        "field FiLM/conditioning and to sample the intermediate cocycle field.",
    )
    field_max_tesla: float = Field(
        default=7.0,
        gt=0.0,
        description="Upper end of the log-field axis (Tesla).",
    )
    cocycle_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the compositional cocycle residual "
        "||G(G(x;s,t);t,u) - G(x;s,u)||_1 (0 disables; the arm reduces to the "
        "cross_field baseline). Curriculum target: 'cocycle_consistency'.",
    )
    identity_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight of the field-identity residual ||G(x;s,s) - x||_1. "
        "Curriculum target: 'field_identity'.",
    )
    latent_cycle_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight of the latent-cycle identifiability term ||E(R(q,b))-q||_1 "
        "(the coboundary consistency of the encode-once architecture).",
    )
    adversarial_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the hinge adversarial generator term against the "
        "field-conditioned discriminator. Curriculum target: 'adversarial' (ramp "
        "from 0 as an adversarial warm-up to defeat identity collapse).",
    )
    triple_sampling: Literal["uniform_distinct"] = Field(
        default="uniform_distinct",
        description="How the cocycle field triple (s, intermediate m, target t) is "
        "drawn each step. 'uniform_distinct': m ~ U[log field_min, log field_max], "
        "distinct from t.",
    )
    detach_inner: bool = Field(
        default=False,
        description="Stop-gradient the inner map G(x;s,t) in the composite so only "
        "the outer render receives the cocycle gradient (cheaper/steadier, weaker "
        "constraint). A named design decision (report it in provenance).",
    )
    contrast_conditioning: bool = Field(
        default=True,
        description="Condition the renderer FiLM (and the discriminator) on the "
        "contrast id (T1w/T2w/FLAIR).",
    )

    @model_validator(mode="after")
    def _check_field_range(self) -> FieldCocycleConfig:
        if self.field_min_tesla >= self.field_max_tesla:
            raise ValueError(
                "FieldCocycleConfig requires field_min_tesla < field_max_tesla; got "
                f"[{self.field_min_tesla}, {self.field_max_tesla}]."
            )
        if not (self.field_min_tesla <= self.reference_field_tesla <= self.field_max_tesla):
            raise ValueError(
                "FieldCocycleConfig.reference_field_tesla="
                f"{self.reference_field_tesla} must lie within the field range "
                f"[{self.field_min_tesla}, {self.field_max_tesla}]."
            )
        return self


class CrossFieldConfig(StrictSchema):
    """Encode-once / render-anywhere cross-field translation knobs."""

    lambda_latent_cycle: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Weight of the latent-cycle identifiability loss ||E(R(q,b)) - q||_1 (0 disables it)."
        ),
    )
    n_render_fields: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of random target fields per step at which the latent-cycle "
            "constraint is enforced (reserved for multi-render extension)."
        ),
    )
    contrast_conditioning: bool = Field(
        default=True,
        description="Condition the renderer FiLM on the contrast id (T1w/T2w/FLAIR).",
    )


class FieldFlowConfig(StrictSchema):
    """Neural-ODE field-flow knobs (B-3.1).

    Read by ``FieldFlowStrategy`` via ``config.training.field_flow``.
    """

    n_steps: int = Field(
        default=20,
        ge=1,
        description="Integration steps for inference translation along the field axis.",
    )
    solver: Literal["euler", "heun"] = Field(
        default="heun",
        description="ODE integrator for inference ('euler' or 'heun').",
    )
    lambda_straightness: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the transport-cost / straightness penalty mean(||v||^2) "
            "encouraging low-energy paths (0 disables it)."
        ),
    )
    velocity_norm: Literal["l1", "l2"] = Field(
        default="l2",
        description=(
            "Norm of the registered FieldFlowVelocityLoss the strategy invokes "
            "('l2' = flow-matching squared error, 'l1' = robust)."
        ),
    )
    contrast_conditioning: bool = Field(
        default=False,
        description=(
            "Reserved (NOT yet wired): FieldFlowStrategy does not read this, so enabling it "
            "would be a silent no-op (pitfall #15). Must stay False until the velocity field "
            "is actually conditioned on contrast id; setting True raises at config-load."
        ),
    )

    @field_validator("contrast_conditioning")
    @classmethod
    def _reject_unwired_contrast_conditioning(cls, v: bool) -> bool:
        # Anti-#15: an exposed-but-unread knob must fail loudly, not silently no-op. FieldFlow
        # never reads contrast_conditioning, so True is unimplemented — raise instead of degrade.
        if v:
            raise ValueError(
                "FieldFlowConfig.contrast_conditioning is reserved and NOT wired into "
                "FieldFlowStrategy; True would be a silent no-op. Leave it False (or wire the "
                "contrast path first). See pitfall #15."
            )
        return v
