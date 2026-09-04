"""CS-MNO training paradigm sub-section schema.

Located at: ``training.cs_mno`` in v6.0 YAML configs.

CS-MNO = Continuous Space-Filling-Curve Spectral-Local Mamba Neural
Operator. The architecture is a two-branch neural operator:

* **Spectral branch** — FFT, multiply by learnt complex weights on the
  lowest ``modes`` Fourier coefficients, IFFT. This is the standard
  FNO truncation; it inherits FNO's universal-approximation guarantee
  on the global low-frequency content.
* **SFC-Mamba branch** — Hilbert-permute the tokens, run a selective
  state-space scan with **physical-arc timesteps**
  ``Δ_t = ‖H(s_t) − H(s_{t-1})‖_Ω``, inverse-permute. The physical
  timestep is the key contribution: it makes the discrete scan a
  consistent quadrature of the continuous-time SSM along the curve,
  which yields Theorem A (resolution invariance) of
  ``TODO/Mamba_NO.md``.

The schema is consumed by ``CSMNOOperator`` (model class registered
with ``training_mode="reconstruction"``). No new training strategy is
required — CS-MNO is an architecture, not a paradigm.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class SpectralConfig(StrictSchema):
    """Spectral (FNO-truncated) branch parameters."""

    modes: int = Field(
        default=16,
        ge=1,
        description=(
            "Number of low-frequency Fourier modes retained per axis "
            "in the spectral branch. The truncation gives a "
            "K^{-s/d} approximation rate on H^s targets (Theorem B)."
        ),
    )
    dim: int = Field(
        default=32,
        ge=4,
        description="Per-layer hidden width of the spectral path.",
    )


class ScanConfig(StrictSchema):
    """SFC-Mamba branch parameters."""

    mode: Literal[
        "hilbert_2d",
        "hilbert_3d",
        "morton_2d",
        "morton_3d",
        "snake_2d",
        "zigzag_2d",
        "raster_2d",
        "raster_3d",
    ] = Field(
        default="hilbert_2d",
        description=(
            "Space-filling curve used to linearise the grid. 'raster_*' "
            "is the LMO baseline; 'hilbert_*' is the proposed default. "
            "Forbidden values raise at config-load time — no silent fallback."
        ),
    )
    use_physical_arc: bool = Field(
        default=True,
        description=(
            "If true, Δ_t per token is the physical arc length between "
            "adjacent SFC samples (Theorem A). If false, the model falls "
            "back to vanilla Mamba's learnt scalar Δ — disabling the "
            "resolution-invariance guarantee. The critical ablation."
        ),
    )
    d_state: int = Field(
        default=16,
        ge=1,
        description=(
            "Hidden state dimension of the selective SSM. Standard Mamba "
            "default is 16; larger values trade memory for capacity."
        ),
    )


class ArchitectureConfig(StrictSchema):
    """CS-MNO stack parameters (depth + lifting/projection widths)."""

    n_layers: int = Field(
        default=4,
        ge=1,
        description="Number of CSMNOLayer stacks between lift and project.",
    )
    lift_dim: int = Field(
        default=32,
        ge=4,
        description=(
            "Output channels of the input-side 1×1 lifting convolution "
            "(matches the FNO pre-processing convention)."
        ),
    )
    activation: Literal["gelu", "relu", "silu"] = Field(
        default="gelu",
        description="Pointwise nonlinearity between layers.",
    )


class CSMNOTrainingConfigSchema(StrictSchema):
    """CS-MNO training-config sub-section.

    Attached to ``BaseTrainingConfigSchema.cs_mno``. Read by
    ``CSMNOOperator.from_config()`` at model construction time.
    """

    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    scan: ScanConfig = Field(default_factory=ScanConfig)
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)


__all__ = [
    "ArchitectureConfig",
    "CSMNOTrainingConfigSchema",
    "ScanConfig",
    "SpectralConfig",
]
