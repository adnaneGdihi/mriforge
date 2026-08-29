r"""Operator-identification training schema (Proposal 1).

Adds the ``operator_id`` paradigm: Lie-algebraic effective-generator
identification of the composite degradation operator via differentiable
simulation-based inference.

Two objects are exported:

* :class:`OperatorIDConfig` — the nested ``training.operator_id`` block
  (also referenced by :class:`mriforge.config.schemas.training.base.TrainingStrategyConfigSchema`).
* :class:`TrainingConfigOperatorID` — the paradigm-specific full training
  config, registered in the ``create_training_config`` dispatch table on
  ``training_mode == "operator_id"``.

The addition is purely additive: every existing schema is unchanged and
the new block defaults to ``None`` on the shared training schema, so all
prior v6.0 YAMLs continue to validate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import BaseTrainingConfigSchema

# Mirrors SUPPORTED_BCH_ORDERS in
# mriforge.infrastructure.physics.magnus_exponential — kept in sync with the
# ``bch_order_supported`` Tier-1 audit check.
SUPPORTED_BCH_ORDERS: tuple[int, ...] = (1, 2, 3)


class OperatorIDConfig(BaseModel):
    r"""Configuration for Lie-algebraic operator identification.

    Located at ``training.operator_id``; consumed by
    ``OperatorIdBCHTrainingStrategy`` and the ``bch_operator_conditioner``
    model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    mode_dictionary: tuple[str, ...] = Field(
        default=(
            "motion_rigid",
            "bias_field_multiplicative",
            "b0_phase",
            "gibbs_truncation",
            "rician_noise",
        ),
        description=(
            "Registered degradation modes forming the operator basis. Each "
            "name must resolve to a generator adapter in "
            "infrastructure.physics.degradation_generators (deterministic "
            "modes enter exp(Ω); noise modes enter the covariance Σ)."
        ),
    )
    bch_order: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Baker-Campbell-Hausdorff / Magnus truncation order (1, 2 or 3).",
    )
    krylov_dim: int = Field(
        default=30,
        ge=1,
        description="Arnoldi/Lanczos subspace size for the matrix-free exp(Ω) action.",
    )
    covariance_rank: int = Field(
        default=8,
        ge=1,
        description="Rank r of the low-rank covariance term U Uᴴ.",
    )
    mmd_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight λ on the unpaired pushforward-MMD term.",
    )
    soft_truncation_tau: float = Field(
        default=50.0,
        gt=0.0,
        description="τ for the Gibbs soft-projection surrogate (sharpness of the k-space cutoff).",
    )
    contrast_field_conditioning: bool = Field(
        default=True,
        description="Condition the operator on (contrast, field) coordinates u=(c, β).",
    )
    num_contrasts: int = Field(
        default=3,
        ge=1,
        description="Cardinality of the contrast vocabulary (e.g. T1w/T2w/FLAIR ⇒ 3).",
    )
    rho_bins: int = Field(
        default=16,
        ge=1,
        description="Number of k-space radial bins for the noise PSD rho.",
    )
    # SOTA plan T3: calibrated operator posterior (replaces blind JSMoCo).
    operator_posterior: Literal["none", "laplace"] = Field(
        default="none",
        description=(
            "Uncertainty posterior over the BCH operator severities. 'laplace' "
            "computes Σ=(H+λ₀I)⁻¹ at the MAP and reports the per-mode posterior "
            "std (operator_posterior.laplace_posterior). 'none' = point estimate. "
            "A meaningful posterior REQUIRES complex multi-coil k-space "
            "(identifiability); enforced by the operator_posterior_requires_complex "
            "audit."
        ),
    )
    posterior_prior_precision: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Isotropic Gaussian-prior precision λ₀ for the Laplace operator "
            "posterior. λ₀→∞ collapses Σ→0 (the BCH point estimate)."
        ),
    )

    @field_validator("mode_dictionary")
    @classmethod
    def _non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) == 0:
            raise ValueError("mode_dictionary must list at least one mode.")
        return v


class TrainingConfigOperatorID(BaseTrainingConfigSchema):
    """Operator-identification (BCH) training configuration."""

    operator_id: OperatorIDConfig = Field(
        default_factory=OperatorIDConfig,
        description="Operator-identification specific configuration.",
    )

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",
        frozen=True,
    )


__all__ = ["SUPPORTED_BCH_ORDERS", "OperatorIDConfig", "TrainingConfigOperatorID"]
