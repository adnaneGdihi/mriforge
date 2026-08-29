"""Phase-2 / Phase-3 schemas for the VF advanced campaign.

Two paradigms covered, both extensions of the existing VF / diffusion
strategies:

* :class:`TrainingConfigIBVF` — Information-Bottleneck VF via InfoNCE
  on (navigator, motion). Adds two MI weights and a critic-config
  sub-block to the existing VF schema.
* :class:`TrainingConfigTwinDPS` — Twin-Likelihood DPS sampling with
  data-consistency + phase-stego marker score guidance. Extends
  :class:`TrainingConfigDiffusion`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseTrainingConfigSchema
from .diffusion import TrainingConfigDiffusion

# ----------------------------------------------------------------------
# Phase 2 — IB-VF / InfoNCE.
# ----------------------------------------------------------------------


class IBVFConfig(BaseModel):
    """IB-VF hyperparameters (training.ib_vf block)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    beta_plus: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight on the I(z_m; theta_hat) maximisation term.",
    )
    beta_minus: float = Field(
        default=0.1,
        gt=0.0,
        description="Weight on the I(z_m; x) minimisation term (squeeze nuisance).",
    )
    infonce_temperature: float = Field(default=0.1, gt=0.0)
    infonce_negatives_K: int = Field(
        default=32,
        ge=8,
        description=(
            "Effective negatives per positive. Tier-1 check requires "
            "data.batch_size * gradient_accumulation_steps + memory_bank_size >= K."
        ),
    )
    memory_bank_size: int = Field(
        default=0,
        ge=0,
        description="MoCo bank capacity. 0 disables.",
    )
    critic_hidden_dim: int = Field(default=128, ge=16)
    nav_encoder_depth: int = Field(default=4, ge=2)
    nav_latent_dim: int = Field(default=64, ge=8)
    navigator_source: Literal["dc_kspace", "central_line"] = Field(
        default="dc_kspace",
        description=(
            "Where the strategy reads the navigator from. dc_kspace "
            "uses y(0, t); central_line uses y(k=0, t) (equivalent for "
            "Cartesian, distinct for non-Cartesian). Tier-1 check "
            "rejects anything else."
        ),
    )


class TrainingConfigIBVF(BaseTrainingConfigSchema):
    """Top-level training config when training_mode == 'ib_vf'."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    ib_vf: IBVFConfig = Field(default_factory=IBVFConfig)


# ----------------------------------------------------------------------
# Phase 3 — Twin-Likelihood DPS.
# ----------------------------------------------------------------------


class TwinDPSConfig(BaseModel):
    """Twin-Likelihood DPS hyperparameters (training.twin_dps block)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_y: float = Field(default=1.0, ge=0.0)
    lambda_M: float = Field(default=1.0, ge=0.0)
    sigma_M: float = Field(default=0.05, gt=0.0)
    marker_template_path: Path = Field(
        ...,
        description="Path to the marker template tensor (complex .pt artefact).",
    )
    phase_stego_basis: Literal["fourier", "wavelet"] = Field(default="fourier")
    guidance_callback_interval: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _bounded_guidance_scales(self) -> TwinDPSConfig:
        for name, val in (("lambda_y", self.lambda_y), ("lambda_M", self.lambda_M)):
            if val > 1e3:
                raise ValueError(f"twin_dps.{name}={val} exceeds the safe upper bound 1e3.")
        return self


class TrainingConfigTwinDPS(TrainingConfigDiffusion):
    """Top-level training config when training_mode == 'twin_dps'.

    Inherits the full diffusion schema (timesteps, noise_schedule,
    prediction_type, sampler, etc.) and adds the twin-likelihood block.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    twin_dps: TwinDPSConfig


__all__ = [
    "IBVFConfig",
    "TrainingConfigIBVF",
    "TrainingConfigTwinDPS",
    "TwinDPSConfig",
]
