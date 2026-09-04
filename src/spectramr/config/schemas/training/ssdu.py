"""SSDU (self-supervised reconstruction) training sub-schema.

Declarative config for ``training_mode: self_supervised_reconstruction`` and the
``robust_ssdu`` (Noisier2Noise) variant. Frozen Pydantic v2 model,
``extra="forbid"``, mounted on ``TrainingStrategyConfigSchema`` as an optional
sub-block (mirrors the v6.1 pattern used by GeoMambaULFTrainingConfigSchema).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SSDUTrainingConfigSchema(BaseModel):
    """Parameters for the SSDU Lambda/Theta k-space split (+ Robust SSDU)."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    theta_fraction: float = Field(
        default=0.4,
        gt=0.0,
        lt=1.0,
        description="Fraction of acquired k-space points held out into Theta (loss set).",
    )
    num_masks: int = Field(
        default=1,
        ge=1,
        description="Number of independent (Lambda,Theta) splits per example (multi-mask SSDU).",
    )
    split_seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed for reproducible Lambda/Theta splits.",
    )

    # --- Robust SSDU (Noisier2Noise, Millard & Chiew 2024) -----------------
    noisier2noise_correction: bool = Field(
        default=False,
        description=(
            "Enable Robust SSDU: add synthetic complex-Gaussian noise of std "
            "noise_std_estimate to the network input (noisier), keep the held-out "
            "Theta loss target at the original (less-noisy) measurement — the "
            "Noisier2Noise principle. Selected by the 'robust_ssdu' strategy key."
        ),
    )
    noise_std_estimate: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "k-space noise σ for the Noisier2Noise correction. Required when "
            "noisier2noise_correction is true (pitfall #15). At σ→0 the correction "
            "reduces to vanilla SSDU."
        ),
    )
    noise_model: Literal["gaussian_kspace", "ncchi_magnitude"] = Field(
        default="gaussian_kspace",
        description=(
            "Dataset-driven noise model shared with the T1 keystone "
            "(complex-Gaussian per-coil k-space vs non-central-χ RSS magnitude). "
            "Never hardcoded to 0.3T."
        ),
    )
    n_coils: int = Field(
        default=1,
        ge=1,
        description="Number of coils (degrees of freedom L) for the nc-χ noise model.",
    )

    @model_validator(mode="after")
    def _validate_robust_requires_noise(self) -> SSDUTrainingConfigSchema:
        if self.noisier2noise_correction and self.noise_std_estimate is None:
            raise ValueError(
                "ssdu.noisier2noise_correction=true requires noise_std_estimate "
                "(the k-space noise σ); none was set (pitfall #15: the knob must "
                "be wired)."
            )
        return self
