"""Minimal-settings loader for per-paradigm smoke tests.

Maps a strategy key (from ``STRATEGY_CLASS_PATHS``) to a
``TrainingSettings`` object built from a small YAML fixture in
``tests/_fixtures/minimal_settings/<family>.yaml``.

The mapping from key → family is intentionally broad: many keys share a
concrete strategy class (e.g. all reconstruction aliases map to
``ReconstructionTrainingStrategy``) so a single minimal YAML covers all of
them.

Usage::

    from tests.utils.minimal_settings import minimal_settings_for

    settings = minimal_settings_for("reconstruction")   # returns TrainingSettings
    settings = minimal_settings_for("unknown_key")       # raises SettingsNotFound

Raises:
    SettingsNotFound: When no fixture family covers the requested key.
    yaml.YAMLError / pydantic.ValidationError: Propagated transparently
        so callers can diagnose bad YAML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mriforge.config.settings import TrainingSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixture root
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent.parent / "_fixtures" / "minimal_settings"


# ---------------------------------------------------------------------------
# Strategy-key → fixture-family mapping
# ---------------------------------------------------------------------------
# Each value must be the stem of a YAML file in _FIXTURE_DIR, e.g.
# "reconstruction" → reconstruction.yaml.
#
# Keys that share a concrete strategy class are grouped under one family.
# When a new specialised family YAML is authored, add an entry here.
# Keys that have NO fixture yet map to None and trigger SettingsNotFound.

_KEY_TO_FAMILY: dict[str, str | None] = {
    # ------------------------------------------------------------------ #
    # Reconstruction family (ReconstructionTrainingStrategy)
    # ------------------------------------------------------------------ #
    "reconstruction": "reconstruction",
    "swarm": "reconstruction",
    "hybrid": "reconstruction",
    "mesh": "reconstruction",
    "multi_task": "reconstruction",
    "swarm_learning": "reconstruction",
    "dual_task": "reconstruction",
    "federated": "reconstruction",
    "sle_compressed_sensing": "reconstruction",
    "spectral_triple": "reconstruction",
    "tropical_mrf": "reconstruction",
    "sheaf_kspace": "reconstruction",
    "hyperbolic_cardiac": "reconstruction",
    "heisenberg_phase": "reconstruction",
    "hjb_trajectory": "reconstruction",
    "primary_ideal": "reconstruction",
    "frontdoor_federated": "reconstruction",
    "hodge_motion": "reconstruction",
    "koopman_fmri": "reconstruction",
    "sheaf_multi_contrast": "reconstruction",
    "hjb_waveform": "reconstruction",
    "stein_federated": "reconstruction",
    "tropical_quantitative_maps": "reconstruction",
    "frontdoor_scanner": "reconstruction",
    "pnp": "reconstruction",
    "pnp_red": "reconstruction",
    "low_rank_sparse": "reconstruction",
    "qsm_pipeline": "reconstruction",
    # ------------------------------------------------------------------ #
    # GAN family (GANTrainingStrategy)
    # ------------------------------------------------------------------ #
    "gan": "gan",
    # ------------------------------------------------------------------ #
    # Diffusion family (DiffusionTrainingStrategy)
    # ------------------------------------------------------------------ #
    "diffusion": "diffusion",
    "rectified_flow": "diffusion",
    "flow_matching": "diffusion",
    "fisher_rao_flow": "diffusion",
    "resetting_diffusion": "diffusion",
    "levy_diffusion": "diffusion",
    "tissue_diffusion_pretrain": "diffusion",
    "graph_cold_diffusion": "diffusion",
    "cold_diffusion": "diffusion",
    "kspace_cold_diffusion": "diffusion",
    # ------------------------------------------------------------------ #
    # VAE family (VAETrainingStrategy / VQVAETrainingStrategy)
    # ------------------------------------------------------------------ #
    "vae": "vae",
    "vqvae": "vae",
    "disentangled_vae": "vae",
    # ------------------------------------------------------------------ #
    # Physics-driven family (PhysicsDrivenTrainingStrategy)
    # ------------------------------------------------------------------ #
    "physics_driven": "physics_driven",
    # ------------------------------------------------------------------ #
    # PINN family (ConcretePINNSensitivityStrategy)
    # ------------------------------------------------------------------ #
    "pinn": "pinn",
    # ------------------------------------------------------------------ #
    # CycleBloch family (CycleBlochStrategy)
    # ------------------------------------------------------------------ #
    "cycle_bloch": "cycle_bloch",
    "symplectic_bloch": "cycle_bloch",
    # ------------------------------------------------------------------ #
    # MAE/SSL family (MAEPretrainingStrategy)
    # ------------------------------------------------------------------ #
    "mae": "mae",
    "ssl": "mae",
    # ------------------------------------------------------------------ #
    # Specialised strategies — no fixture yet; skip on cluster until YAML
    # is authored and validated.
    # ------------------------------------------------------------------ #
    "domain_adaptation": None,
    "volumetric": None,
    "ttt": None,
    "meta_learning": None,
    "disentangled": None,
    "masked": None,
    "padnet": None,
    "noise_to_noise": None,
    "n2n": None,
    "diff_siren": None,
    "virtual_fiducial": None,
    "distillation": None,
    "motion_meta": None,
    "tto": None,
    "test_time_optimization": None,
    "vf_admm": None,
    "pma_varnet": None,
    "multi": None,
    "test_time_adaptation": None,
    "3d_generation": None,
    "adaptation": None,
    "disentangled_diffusion": None,
    "guided_sr": None,
    "geomamba_ulf": None,
    "hssc": None,
    "dtn2s": None,
    "privileged_learning": None,
    "privileged": None,
    "slice_to_volume": None,
    "2d_to_3d": None,
    "calibration": None,
    "validation_badge": None,
    "learnable_acquisition": None,
    "loupe": None,
    "pilot": None,
    "active_acquisition": None,
    "bald": None,
    "multi_param_mapping": None,
    "multi_contrast_contrastive": None,
    "edm": None,
    "concomitant_aware_recon": None,
    "caur": None,
    "xfield_fm": None,
    "scas": None,
    "federated_dp_conformal": None,
    "kspace_inr": None,
    "bloch_consistent_denoising": None,
    "girf_aware": None,
    "inverse_bloch_phase": None,
    "qspace_diffusion": None,
    "adversarial_robustness": None,
    "stochastic_interpolants": None,
    "jepa": None,
    "mri_slam": None,
    "noise_adaptive_score": None,
    "diffeomorphic_recon": None,
    "field_probe_coupled": None,
    "spin_sde": None,
    "coord_kspace_gen": None,
    "bloch_bottleneck": None,
    "cycle_bloch_digital_twin": None,
    "cross_contrast_kspace_diffusion": None,
    "synthetic_pathology_aug": None,
    "physics_equivariant_ssl": None,
    "phys_eq_ssl": None,
    "ib_active_acquisition": None,
    "ib_bald": None,
    "score_field_tomography": None,
    "bloch_equivariant_translation": None,
    "riemannian_bloch_diffusion": None,
    "adaptive_sfc_hssc": None,
    "conformal_diffusion_recon": None,
    "beltrami_motion_correction": None,
    "cortical_conformal_recon": None,
    "teichmuller_cold_diffusion": None,
    "spatiotemporal_adaptive_sfc_recon": None,
    "beltrami_epi_distortion": None,
    "cortical_conformal_fmri_recon": None,
    "riemannian_dfc_diffusion": None,
    "hrf_manifold_diffusion": None,
    "spatiotemporal_mrf_recon": None,
    "riemannian_mrf_diffusion": None,
    "conformal_mrf_dictless_recon": None,
    "crlb_mrf_pulse_design": None,
    "cross_scanner_mrf_harmonisation": None,
    "ib_vf": None,
    "twin_dps": None,
    "corruption_calibration": None,
    "paired_synthesis": None,
    "data_efficiency_harness": None,
}


class SettingsNotFound(Exception):
    """Raised when no minimal-settings fixture covers the requested key."""


def minimal_settings_for(key: str) -> "TrainingSettings":
    """Return a ``TrainingSettings`` for *key* loaded from a minimal fixture YAML.

    Args:
        key: A strategy key from ``STRATEGY_CLASS_PATHS`` (e.g. ``"reconstruction"``).

    Returns:
        ``TrainingSettings`` instance loaded from the corresponding fixture YAML.

    Raises:
        SettingsNotFound: No fixture family is mapped for *key*.
        FileNotFoundError: The fixture YAML file is missing on disk.
        yaml.YAMLError | pydantic.ValidationError: Propagated from the loader.
    """
    family = _KEY_TO_FAMILY.get(key)

    if family is None:
        raise SettingsNotFound(
            f"No minimal-settings fixture mapped for strategy key '{key}'. "
            "Author a YAML under tests/_fixtures/minimal_settings/<family>.yaml "
            "and add an entry to _KEY_TO_FAMILY in tests/utils/minimal_settings.py."
        )

    yaml_path = _FIXTURE_DIR / f"{family}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Minimal-settings fixture file not found: {yaml_path}. "
            f"Expected at tests/_fixtures/minimal_settings/{family}.yaml."
        )

    from mriforge.config.settings import TrainingSettings  # noqa: PLC0415

    logger.debug("Loading minimal settings for key=%r from %s", key, yaml_path)
    return TrainingSettings.from_yaml(str(yaml_path))


def family_for(key: str) -> str | None:
    """Return the fixture-family name for *key*, or None if unregistered.

    Useful for test parametrization logic that needs to distinguish
    "missing mapping" (key not in dict) from "no fixture yet" (mapped to None).
    """
    return _KEY_TO_FAMILY.get(key, "UNKNOWN_KEY")


def covered_keys() -> list[str]:
    """Return sorted list of strategy keys that have a non-None fixture mapping."""
    return sorted(k for k, v in _KEY_TO_FAMILY.items() if v is not None)


def uncovered_keys() -> list[str]:
    """Return sorted list of strategy keys with no fixture yet (mapped to None)."""
    return sorted(k for k, v in _KEY_TO_FAMILY.items() if v is None)
