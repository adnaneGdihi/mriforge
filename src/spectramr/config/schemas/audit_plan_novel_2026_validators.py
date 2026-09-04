"""Tier-1 / Tier-2 audit checks for the SFC/conformal + fMRI/MRF arms.

Covers the eight gaps from the cross-plan review:

* SFC §1 — adaptive_sfc_paired_with_compatible_model
* SFC §2 — conformal_dc_requires_jacobian_field
* SFC §5 — cortical_recon_requires_flatten_grid
* SFC §6 — teichmuller_schedule_in_cold_diffusion_only
* fMRI §2 — epi_phase_encode_direction_required (Tier-1)
* MRF §1  — mrf_spiral_rotation_metadata_required (Tier-1)
* MRF §4  — mrf_pulse_sar_compliance (Tier-2)
* MRF §5  — mrf_harmonisation_phantom_calibration_required (Tier-2)

Each rule follows the standard validator signature
``(config_dict) -> list[str]`` and is registered through
:func:`register_audit_plan_novel_2026_validators`.

References:
    [9]  Andersson et al., "How to correct susceptibility distortions",
         *NeuroImage*, 20(2), 2003.
    [21] Buonincontri et al., "Multi-site repeatability of MRF",
         *NeuroImage*, 195, 2019.
"""

from __future__ import annotations

from typing import Any

from spectramr.config.schemas.validator_registry import (
    ValidationRule,
    ValidatorRegistry,
)

# --------------------------------------------------------------------- #
# SFC §1 — adaptive_sfc_hssc requires a model that accepts sfc_indices.
# --------------------------------------------------------------------- #

_SFC_INDEX_AWARE_MODELS = {
    "conformal_geomamba_ulf",  # consumes adaptive SFC implicitly
}


def _expose_flag(config: dict[str, Any], leaf: str) -> bool:
    """Read a `data.expose.<leaf>` flag under either spelling.

    The flat `data.expose_<leaf>` spelling is a retired rename record: arms in
    `experiments/inprogress/` were drained to `data.expose.<leaf>` and the flat
    form now raises at load. Reading only the flat name made these validators
    warn on arms that DO set the flag, and told the author to set a spelling
    that no longer loads. The other corpus roots are un-drained, so both
    spellings still have to resolve.
    """
    data_cfg = config.get("data") or {}
    expose = data_cfg.get("expose")
    if isinstance(expose, dict) and leaf in expose:
        return bool(expose[leaf])
    return bool(data_cfg.get(f"expose_{leaf}", False))


def _validate_adaptive_sfc_paired_with_compatible_model(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "adaptive_sfc_hssc":
        return []
    model_type = config.get("model", {}).get("model_type")
    # Warn (not error) — strategy passes sfc_indices through **kwargs and
    # models that ignore it are still trained on the L1 objective with the
    # Beltrami head as a side regulariser. Surfacing the mismatch helps
    # users notice they're not exploiting the adaptive ordering.
    if model_type not in _SFC_INDEX_AWARE_MODELS:
        return [
            f"adaptive_sfc_hssc was paired with model {model_type!r}, which does "
            "not declare adaptive-SFC compatibility. The Beltrami head will still "
            "train but the model will ignore the learned ordering. Consider "
            "conformal_geomamba_ulf or extending your model's forward signature "
            "to accept sfc_indices=…"
        ]
    return []


# --------------------------------------------------------------------- #
# SFC §2 — ConformalDataConsistency needs conformal_jacobian per batch.
# --------------------------------------------------------------------- #


def _validate_conformal_dc_requires_jacobian_field(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "conformal_diffusion_recon":
        return []
    # The strategy gracefully falls back to identity Jacobian + plain DC
    # when the batch lacks the field, so this is a warning rather than an
    # error. The user is meant to either populate the field via a transform
    # or accept the audit warning explicitly.
    expose = _expose_flag(config, "conformal_jacobian")
    if not expose:
        return [
            "conformal_diffusion_recon expects batch['conformal_jacobian'] to be "
            "populated by the data pipeline. Set data.expose.conformal_jacobian: "
            "true once the corresponding transform is wired; until then the "
            "strategy will fall back to an identity Jacobian (equivalent to "
            "HardDataConsistency)."
        ]
    return []


# --------------------------------------------------------------------- #
# SFC §5 — cortical_conformal_recon needs cortex_flatten_grid per batch.
# --------------------------------------------------------------------- #


def _validate_cortical_recon_requires_flatten_grid(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode not in {
        "cortical_conformal_recon",
        "cortical_conformal_fmri_recon",
    }:
        return []
    expose = _expose_flag(config, "cortex_flatten_grid")
    if not expose:
        return [
            "cortical_conformal_recon expects batch['cortex_flatten_grid'] from a "
            "precomputed conformal flattening service. Set "
            "data.expose.cortex_flatten_grid: true once available, otherwise the "
            "strategy degrades to a vanilla L1 reconstruction with a "
            "Laplacian-difference regulariser."
        ]
    return []


# --------------------------------------------------------------------- #
# SFC §6 — Teichmüller schedule should only be used with cold-diffusion.
# --------------------------------------------------------------------- #


def _validate_teichmuller_schedule_in_cold_diffusion_only(
    config: dict[str, Any],
) -> list[str]:
    losses = config.get("losses") or {}
    image_losses = losses.get("image_losses") or []
    declares = any(
        (entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None))
        == "teichmuller_geodesic"
        for entry in image_losses
    )
    if not declares:
        return []
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode not in {
        "teichmuller_cold_diffusion",
        "cold_diffusion",
    }:
        return [
            f"teichmuller_geodesic loss was declared with training_mode "
            f"{training_mode!r}; the geodesic-schedule regulariser is only "
            "meaningful under cold-diffusion-style training. Either switch to "
            "training_mode: teichmuller_cold_diffusion or drop the loss."
        ]
    return []


# --------------------------------------------------------------------- #
# fMRI §2 — Beltrami EPI distortion requires phase-encode metadata.
# --------------------------------------------------------------------- #


def _validate_epi_phase_encode_direction_required(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "beltrami_epi_distortion":
        return []
    data_cfg = config.get("data") or {}
    pe_axis = data_cfg.get("phase_encode_axis")
    if pe_axis is None:
        return [
            "beltrami_epi_distortion strategy requires data.phase_encode_axis to "
            "be set (-2 for the height axis, -1 for the width axis). Without it "
            "the EPI forward operator silently defaults to -2, which may mismatch "
            "the dataset's acquisition convention."
        ]
    return []


# --------------------------------------------------------------------- #
# MRF §1 — Spiral MRF reconstruction needs per-timepoint rotation schedule.
# --------------------------------------------------------------------- #


def _validate_mrf_spiral_rotation_metadata_required(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "spatiotemporal_mrf_recon":
        return []
    mrf_cfg = config.get("mrf") or {}
    sched = mrf_cfg.get("spiral_rotation_schedule") or mrf_cfg.get("conformal_kspace_map")
    if sched is None:
        return [
            "spatiotemporal_mrf_recon requires mrf.spiral_rotation_schedule (or "
            "mrf.conformal_kspace_map) so the per-timepoint conformal map can "
            "precompute the rotated spiral arm geometry. Add e.g. "
            "mrf: {spiral_rotation_schedule: golden_angle, n_timepoints: 1000}."
        ]
    return []


# --------------------------------------------------------------------- #
# MRF §4 (Tier-2) — designed pulse sequence must respect SAR limits.
# --------------------------------------------------------------------- #


def _validate_mrf_pulse_sar_compliance(config: dict[str, Any]) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "crlb_mrf_pulse_design":
        return []
    mrf_cfg = config.get("mrf") or {}
    sar = mrf_cfg.get("sar_max_w_per_kg")
    gradient = mrf_cfg.get("gradient_slew_rate_t_per_m_per_s")
    if sar is None or gradient is None:
        return [
            "crlb_mrf_pulse_design must declare mrf.sar_max_w_per_kg AND "
            "mrf.gradient_slew_rate_t_per_m_per_s before a designed sequence "
            "can be deployed to a real scanner. Without these the audit cannot "
            "verify SAR compliance and the design output should be considered "
            "simulation-only."
        ]
    return []


# --------------------------------------------------------------------- #
# MRF §5 (Tier-2) — cross-scanner harmonisation must reference phantom data.
# --------------------------------------------------------------------- #


def _validate_mrf_harmonisation_phantom_calibration_required(
    config: dict[str, Any],
) -> list[str]:
    training_mode = config.get("training", {}).get("training_mode")
    if training_mode != "cross_scanner_mrf_harmonisation":
        return []
    mrf_cfg = config.get("mrf") or {}
    calibration = mrf_cfg.get("phantom_calibration_path") or mrf_cfg.get("calibration_dataset")
    if calibration is None:
        return [
            "cross_scanner_mrf_harmonisation requires a NIST/ISMRM phantom "
            "calibration source (mrf.phantom_calibration_path or "
            "mrf.calibration_dataset). Without paired cross-vendor scans of a "
            "reference phantom the learned τ_s cannot be verified to capture "
            "scanner-side variation rather than residual subject-side variation."
        ]
    return []


# --------------------------------------------------------------------- #
# Registry hook
# --------------------------------------------------------------------- #


_RULES = [
    ValidationRule(
        name="adaptive_sfc_paired_with_compatible_model",
        category="audit_plan_novel_2026",
        description=(
            "adaptive_sfc_hssc benefits from a model that can consume the "
            "learned sfc_indices (SFC §1)."
        ),
        validator_fn=_validate_adaptive_sfc_paired_with_compatible_model,
        severity="warning",
        applies_to=["adaptive_sfc_hssc"],
    ),
    ValidationRule(
        name="conformal_dc_requires_jacobian_field",
        category="audit_plan_novel_2026",
        description=(
            "ConformalDataConsistency needs batch['conformal_jacobian'] for the "
            "full forward weighting (SFC §2)."
        ),
        validator_fn=_validate_conformal_dc_requires_jacobian_field,
        severity="warning",
        applies_to=["conformal_diffusion_recon"],
    ),
    ValidationRule(
        name="cortical_recon_requires_flatten_grid",
        category="audit_plan_novel_2026",
        description=(
            "Cortical-conformal reconstruction needs a precomputed cortical "
            "flattening grid in the batch (SFC §5 / fMRI §3)."
        ),
        validator_fn=_validate_cortical_recon_requires_flatten_grid,
        severity="warning",
        applies_to=[
            "cortical_conformal_recon",
            "cortical_conformal_fmri_recon",
        ],
    ),
    ValidationRule(
        name="teichmuller_schedule_in_cold_diffusion_only",
        category="audit_plan_novel_2026",
        description=(
            "teichmuller_geodesic loss is the SFC-schedule regulariser and is "
            "meaningless outside cold-diffusion training (SFC §6)."
        ),
        validator_fn=_validate_teichmuller_schedule_in_cold_diffusion_only,
        severity="warning",
        # Triggers on every config; the validator's own body short-circuits
        # when the loss isn't declared, so the cost is negligible.
        applies_to=None,
    ),
    ValidationRule(
        name="epi_phase_encode_direction_required",
        category="audit_plan_novel_2026",
        description=(
            "Beltrami EPI distortion strategies need an explicit phase-encode "
            "axis in the data block (fMRI §2)."
        ),
        validator_fn=_validate_epi_phase_encode_direction_required,
        severity="error",
        applies_to=["beltrami_epi_distortion"],
    ),
    ValidationRule(
        name="mrf_spiral_rotation_metadata_required",
        category="audit_plan_novel_2026",
        description=(
            "Spiral MRF reconstruction needs the per-timepoint rotation "
            "schedule so the conformal k-space map is well-defined (MRF §1)."
        ),
        validator_fn=_validate_mrf_spiral_rotation_metadata_required,
        severity="error",
        applies_to=["spatiotemporal_mrf_recon"],
    ),
    ValidationRule(
        name="mrf_pulse_sar_compliance",
        category="audit_plan_novel_2026",
        description=(
            "Designed pulse sequences cannot ship without SAR and slew-rate "
            "ceilings — Tier-2 safety check (MRF §4)."
        ),
        validator_fn=_validate_mrf_pulse_sar_compliance,
        severity="error",
        applies_to=["crlb_mrf_pulse_design"],
    ),
    ValidationRule(
        name="mrf_harmonisation_phantom_calibration_required",
        category="audit_plan_novel_2026",
        description=(
            "Cross-scanner harmonisation must reference a NIST-style phantom "
            "calibration source (MRF §5)."
        ),
        validator_fn=_validate_mrf_harmonisation_phantom_calibration_required,
        severity="error",
        applies_to=["cross_scanner_mrf_harmonisation"],
    ),
]


def register_audit_plan_novel_2026_validators(registry: ValidatorRegistry) -> None:
    """Register every rule defined in this module."""
    for rule in _RULES:
        registry.register(rule)


__all__ = ["register_audit_plan_novel_2026_validators"]
