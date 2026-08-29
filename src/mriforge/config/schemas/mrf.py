"""Pydantic schema for the optional top-level ``mrf:`` configuration block.

Hosts the metadata that the MRF strategies of
``TODO/audit_plan_novel_fmri.md`` depend on but that does not fit
under ``data``, ``model``, or ``training``:

* MRF §1 — ``spiral_rotation_schedule`` / ``conformal_kspace_map`` /
  ``n_timepoints`` for the per-timepoint conformal map.
* MRF §4 — ``sar_max_w_per_kg`` / ``gradient_slew_rate_t_per_m_per_s``
  for the Tier-2 SAR-compliance audit.
* MRF §5 — ``phantom_calibration_path`` /
  ``calibration_dataset`` for cross-scanner harmonisation.

The block is opt-in; YAMLs that don't set ``mrf:`` continue to load
unchanged (the parent ``TrainingSettings`` field defaults to
``None``). The corresponding audit validators in
:mod:`mriforge.config.schemas.audit_plan_novel_2026_validators` consume
this block when the strategy requires it.

References:
    [1] D. Ma et al., "Magnetic resonance fingerprinting", *Nature*,
        495, 2013.
    [18] B. Zhao et al., "Optimal experiment design for MRF: CRLB
         meets spin dynamics", *IEEE TMI*, 38(3), 2019.
    [21] Buonincontri et al., "Multi-site repeatability of MRF",
         *NeuroImage*, 195, 2019.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MRFConfigSchema(BaseModel):
    """Top-level ``mrf:`` block — see module docstring."""

    model_config = {"extra": "forbid", "frozen": True, "protected_namespaces": ()}

    # § 1 — spiral / conformal trajectory metadata.
    spiral_rotation_schedule: Literal["golden_angle", "linear", "random"] | None = Field(
        default=None,
        description=(
            "Rotation schedule of the spiral arm across timepoints. "
            "``golden_angle`` matches the standard FISP-MRF convention; "
            "``linear`` is a simple Δθ ramp; ``random`` samples from "
            "U(0, 2π) per timepoint."
        ),
    )
    conformal_kspace_map: str | None = Field(
        default=None,
        description=(
            "Optional path or registry name of a precomputed conformal "
            "Riemann map from the spiral region to the unit disk. When "
            "absent the rotation_schedule is enough to derive the map "
            "lazily for the standard rotated-spiral case."
        ),
    )
    n_timepoints: int | None = Field(
        default=None,
        ge=1,
        description="Number of MRF readouts per acquisition.",
    )

    # § 4 — Pulse-design SAR / hardware bounds.
    sar_max_w_per_kg: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Maximum permissible specific-absorption rate (W/kg). The "
            "Tier-2 audit ``mrf_pulse_sar_compliance`` rejects designed "
            "sequences that exceed this bound."
        ),
    )
    gradient_slew_rate_t_per_m_per_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Maximum gradient slew rate (T/m/s). Bounds the per-TR change in the readout direction."
        ),
    )

    # § 5 — Cross-scanner harmonisation calibration.
    phantom_calibration_path: str | None = Field(
        default=None,
        description=(
            "Filesystem path to a NIST/ISMRM phantom calibration dataset "
            "(paired cross-vendor fingerprints). Required by the Tier-2 "
            "audit ``mrf_harmonisation_phantom_calibration_required``."
        ),
    )
    calibration_dataset: str | None = Field(
        default=None,
        description=(
            "Alternative registry-name reference to a calibration "
            "dataset; satisfied either by ``phantom_calibration_path`` "
            "or by this key."
        ),
    )


__all__ = ["MRFConfigSchema"]
