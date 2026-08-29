"""Hamiltonian-NODE acquisition-learning configuration schema (B-4 / Bundle P7).

Defines the ``training.hamiltonian_acquisition`` sub-block read by
``HamiltonianAcquisitionStrategy``. The block parameterises the learnable
potential :math:`V_\\theta`, the symplectic integrator, the hardware bounds, and
the two auxiliary-loss weights.

Mounting note (intentional, not a silent fallback):
``BaseTrainingConfigSchema`` is ``extra="ignore"`` and
``TrainingStrategyConfigSchema`` is ``extra="allow"``, so a YAML carrying
``training.hamiltonian_acquisition`` validates without the central
``schemas/training/__init__.py`` dispatch table being edited. The strategy
coerces the raw mapping into :class:`HamiltonianAcquisitionConfig` at setup time
(mirroring how ``TTOTrainingStrategy`` coerces ``training.tto``), giving typed
attribute access downstream. To make the block a *typed* field on the base
schema, add ``hamiltonian_acquisition`` to ``TrainingStrategyConfigSchema`` and
register the mode in ``_MODE_DISPATCH`` — both are shared files and out of scope
for this component.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["HamiltonianAcquisitionConfig"]


class HamiltonianAcquisitionConfig(BaseModel):
    """Typed parameters for Hamiltonian-NODE trajectory optimisation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- Potential network V_theta ---
    potential_arch: Literal["mlp", "siren", "fourier_features_mlp"] = Field(
        default="siren",
        description="Architecture of the learnable scalar potential V_theta.",
    )
    potential_hidden_dim: int = Field(
        default=128, ge=8, description="Hidden width of the potential network."
    )
    potential_num_layers: int = Field(
        default=4, ge=1, le=16, description="Number of hidden layers in V_theta."
    )

    # --- Symplectic integrator ---
    integrator: Literal["leapfrog", "rk4_symplectic"] = Field(
        default="leapfrog",
        description="Symplectic integrator (energy-conserving).",
    )
    dt_seconds: float = Field(
        default=4e-6,
        gt=0.0,
        description="Gradient-raster integration step Δt (seconds).",
    )
    T_total_seconds: float = Field(
        default=50e-3,
        gt=0.0,
        description="Total readout window T (seconds).",
    )
    num_shots: int = Field(default=1, ge=1, description="Number of independent trajectory arms.")
    samples_per_shot: int = Field(
        default=256,
        ge=2,
        description="Stored knots per shot (integration steps = this - 1).",
    )

    # --- Hardware bounds (clinical scanner envelope) ---
    G_max_mT_per_m: float = Field(
        default=40.0,
        gt=0.0,
        le=80.0,
        description="Gradient-amplitude bound (mT/m); >80 is beyond clinical scanners.",
    )
    S_max_T_per_m_per_s: float = Field(
        default=200.0,
        gt=0.0,
        le=500.0,
        description="Slew-rate bound (T/m/s); >500 is beyond clinical scanners.",
    )

    # --- Loss weights ---
    energy_conservation_weight: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight on the Hamiltonian energy-conservation penalty.",
    )
    hardware_penalty_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight on the gradient/slew hardware-compliance penalty.",
    )
    num_trajectories_per_batch: int = Field(
        default=1,
        ge=1,
        description="How many trajectory families to integrate per training step.",
    )

    # --- Physical constants for normalised→physical conversion ---
    gamma_hz_per_t: float = Field(
        default=42.577478e6,
        gt=0.0,
        description="Gyromagnetic ratio gamma/2pi (Hz/T); proton default.",
    )
    fov_m: float = Field(
        default=0.22,
        gt=0.0,
        description="Field of view (m) for normalised-k-space → physical-gradient scaling.",
    )

    @model_validator(mode="after")
    def _at_least_one_aux_active(self) -> HamiltonianAcquisitionConfig:
        """A trajectory arm with both auxiliary weights at zero is suspicious.

        The recon loss can still drive learning, but a Hamiltonian-acquisition
        arm with no energy term AND no hardware term has discarded the entire
        physics rationale of the method — flag it loudly rather than silently
        train an unconstrained trajectory (CLAUDE.md #9/#10).
        """
        if self.energy_conservation_weight == 0.0 and self.hardware_penalty_weight == 0.0:
            raise ValueError(
                "Both energy_conservation_weight and hardware_penalty_weight are 0. "
                "A Hamiltonian-acquisition arm needs at least one physics penalty "
                "active; otherwise the trajectory is unconstrained and the method's "
                "hardware-compliance guarantee is void."
            )
        return self
