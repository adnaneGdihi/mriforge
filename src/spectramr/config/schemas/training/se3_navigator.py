r"""Schema for the SE(3)-equivariant DC-navigator (B-3 of the VF campaign).

Mounted under ``training_mode == 'se3_equivariant_navigator'`` (the lead wires
the dispatch entry in ``config/schemas/training/__init__.py``). Carries the
hyperparameters of :class:`spectramr.models.navigation.se3_equivariant_dc_navigator.SE3EquivariantDCNavigator`
and the IB-InfoNCE navigator-latent weights, plus the Nyquist / physiological
validators that the Tier-1 audit checks describe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseTrainingConfigSchema


class SE3NavigatorConfig(BaseModel):
    """SE(3)-navigator hyperparameters (``training.se3_equivariant_navigator`` block)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_mamba_layers: int = Field(default=4, ge=1, le=16)
    d_state: int = Field(default=16, ge=4)
    d_conv: int = Field(default=4, ge=2)
    dc_navigator_sample_rate_Hz: float = Field(  # noqa: N815 — unit-bearing YAML key
        default=100.0,
        gt=0.0,
        description="Navigator temporal sampling rate (Hz).",
    )
    bio_harmonic_freqs_Hz: tuple[float, ...] = Field(  # noqa: N815 — unit-bearing YAML key
        default=(0.3, 0.6, 1.2, 2.4),
        description="Physiological resonance frequencies (Hz) for eigenvalue init.",
    )
    ib_beta_plus: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight on I(z_m; motion) maximisation (information bottleneck).",
    )
    ib_beta_minus: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight on I(z_m; anatomy) minimisation (squeeze nuisance).",
    )
    infonce_temperature: float = Field(default=0.1, gt=0.0)
    group: Literal["SO3", "SE3"] = Field(default="SE3")
    lambda_se3_equivariance: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight on the SE(3)-equivariance-defect monitoring loss "
        "(read by SE3EquivariantNavigatorStrategy; lives here, not on the "
        "extra='ignore' reconstruction-loss block where it was silently dropped).",
    )

    @model_validator(mode="after")
    def _validate_physiology_and_nyquist(self) -> SE3NavigatorConfig:
        if not self.bio_harmonic_freqs_Hz:
            raise ValueError("bio_harmonic_freqs_Hz must be non-empty.")
        for f in self.bio_harmonic_freqs_Hz:
            if f <= 0:
                raise ValueError(f"bio_harmonic_freqs_Hz must be strictly positive; got {f}.")
            if not (0.1 <= f <= 5.0):
                # physiological band guard (Tier-1 'bio_harmonic_freqs_physiological')
                raise ValueError(
                    f"bio_harmonic_freqs_Hz entry {f} Hz outside the physiological "
                    "band [0.1, 5.0] Hz."
                )
        nyquist = self.dc_navigator_sample_rate_Hz / 2.0
        if max(self.bio_harmonic_freqs_Hz) >= nyquist:
            # Tier-1 'dc_navigator_nyquist_satisfied'.
            raise ValueError(
                f"dc_navigator_sample_rate_Hz={self.dc_navigator_sample_rate_Hz} "
                f"violates Nyquist for max bio-harmonic "
                f"{max(self.bio_harmonic_freqs_Hz)} Hz (needs > "
                f"{2 * max(self.bio_harmonic_freqs_Hz)} Hz)."
            )
        return self


class TrainingConfigSE3Navigator(BaseTrainingConfigSchema):
    """Top-level training config when ``training_mode == 'se3_equivariant_navigator'``."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    se3_equivariant_navigator: SE3NavigatorConfig = Field(default_factory=SE3NavigatorConfig)


__all__ = ["SE3NavigatorConfig", "TrainingConfigSE3Navigator"]
