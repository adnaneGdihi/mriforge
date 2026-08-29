"""Schema for the top-level ``backend_acceleration:`` block (SPECTRA backend).

This is the *hardware execution backend* config — distinct from the existing
``acceleration:`` block (``acceleration.py``), which configures k-space
undersampling. The name is ``backend_acceleration`` precisely to avoid that
collision.

It surfaces the SPECTRA inference accelerator (RISC-V Rocket + WCSR systolic
array + NMAE physics-remap, driven over the RoCC contract) to the framework as
a first-class, audit-validated backend, per SPECTRA implementation plan
Workstream C "Pydantic schema additions". Selecting ``backend == "spectra"``
without this block present is a Tier-0/Tier-1 failure, not a silent default —
the schema-discipline invariant.

YAML shape::

    backend_acceleration:
      backend: spectra
      mesh_rows: 4
      mesh_cols: 4
      lanes: 4
      pe_mode_policy: mixed          # PrecisionPolicy: mixed|fp32|bf16|fp8
      enable_nmae: true
      gridding_kernel_width: 4       # J; drives the A4 gridding accumulator sizing
      oversampling: 2.0              # sigma
      per_channel_scaling: true
      runtime: cosim                 # SpectraRuntime: cosim|rocc
      cosim_bridge_addr: "127.0.0.1:5555"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import PrecisionPolicy, SpectraRuntime


class BackendAccelerationConfigSchema(BaseModel):
    """SPECTRA hardware inference-backend configuration.

    Frozen and ``extra="forbid"``: any unknown key is a hard Tier-0 failure so a
    typo in a hardware knob can never silently fall back to a default. Field
    defaults mirror the published 4x4 / LANES-4 WCSR configuration.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), frozen=True)

    backend: str = Field(
        default="spectra",
        description=(
            "Execution backend selector. Must be 'spectra' to engage this block "
            "(the check_spectra_backend_schema Tier-1 guard rejects any other "
            "value here)."
        ),
    )
    mesh_rows: int = Field(
        default=4,
        ge=1,
        description="Systolic-array rows (MESH_ROWS). 4 matches the published WCSR cluster.",
    )
    mesh_cols: int = Field(
        default=4,
        ge=1,
        description="Systolic-array columns (MESH_COLS).",
    )
    lanes: int = Field(
        default=4,
        ge=1,
        description="DVCP compaction lanes per row (LANES).",
    )
    pe_mode_policy: PrecisionPolicy = Field(
        default=PrecisionPolicy.MIXED,
        description=(
            "Per-layer precision policy for the morphable PE (A7). 'mixed' lets "
            "the compiler pick FP32/BF16/FP8 per layer against a fidelity floor."
        ),
    )
    enable_nmae: bool = Field(
        default=True,
        description=(
            "Enable the NMAE physics-remap path (fft-shift / transpose / "
            "depth-to-space bijective remaps) on the AXI memory boundary."
        ),
    )
    gridding_kernel_width: int = Field(
        default=4,
        ge=1,
        description=(
            "NUFFT gridding kernel width J (A4). Each non-uniform sample scatters "
            "into a J**spatial_dims neighbourhood; check_spectra_gridding_sized "
            "bounds this against the synthesized accumulator depth."
        ),
    )
    oversampling: float = Field(
        default=2.0,
        gt=1.0,
        description="NUFFT grid oversampling factor sigma (A4). Standard KB value is 2.0.",
    )
    per_channel_scaling: bool = Field(
        default=True,
        description=(
            "Use per-channel (vs single global) quantization scales (A7). Required "
            "to cover the dynamic range of MRI feature maps under FP8."
        ),
    )
    runtime: SpectraRuntime = Field(
        default=SpectraRuntime.COSIM,
        description=(
            "Execution runtime. 'cosim' drives the real RTL over the Verilator "
            "TCP bridge (correctness, not performance); 'rocc' targets silicon."
        ),
    )
    cosim_bridge_addr: str | None = Field(
        default=None,
        description=(
            "host:port of the Verilator co-sim MMIO bridge (required only when "
            "runtime == cosim and a real dispatch is performed)."
        ),
    )


__all__ = ["BackendAccelerationConfigSchema"]
