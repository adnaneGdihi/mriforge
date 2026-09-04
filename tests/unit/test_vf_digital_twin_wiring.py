"""Regression guard for the VF digital-twin wiring pass (VF review, 2026-06-02).

Context
-------
The ``virtual_fiducial`` / ``vf_admm`` strategies build their degradation
simulator *only* from ``config.physics.digital_twin`` (see
``simulator_builder.build_simulator_from_config``).  Before this pass, 27 VF
arms shipped **no** ``digital_twin`` block, so they silently fell back to the
schema-default corruption (motion+B0+B1) — identical across every arm,
regardless of each arm's stated hypothesis (undersampling, B1+ mapping, …).

This test asserts every such arm now declares an *enabled* twin with at least
one degradation flag on, and that the degradation matches the arm's physics
axis.  It is data-free (pure schema load) so it runs in the unit tier.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPECTRAMR_SUPPRESS_CLINICAL_WARNING", "1")

import pathlib

import pytest

from spectramr.config.settings import TrainingSettings

VF_DIR = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "inprogress" / "vf"

# Arms wired in the 2026-06-02 pass -> the degradation flag(s) that MUST be on
# (the physics axis the hypothesis is about). One representative flag per arm
# is enough to catch a silent revert to the confounded default.
EXPECTED_AXIS: dict[str, list[str]] = {
    # undersampled reconstruction (acceleration mirrors the stated R)
    "exp_vf_05_csm_boundary_extrap_v2": ["enable_undersampling"],
    "exp_vf_06_vcc_phase_constrained_v2": ["enable_undersampling"],
    "exp_vf_08_deep_dc_forcing_v2": ["enable_undersampling"],
    "exp_vf_09_spatially_variant_reg_v2": ["enable_undersampling"],
    "exp_vf_34_marker_grappa_convolution_v2": ["enable_undersampling"],
    # B0 field
    "exp_vf_04_b0_drift_demodulation_v2": ["enable_b0"],
    "exp_vf_10_bssfp_banding_mitigation_sim_v2": ["enable_b0"],
    "exp_vf_25_epi_geometric_shift_b0_sim_v2": ["enable_b0", "enable_b0_distortion"],
    "exp_vf_26_fid_navigator_b0_sim_v2": ["enable_b0"],
    "exp_vf_29_bssfp_banding_freq_b0_sim_v2": ["enable_b0"],
    "exp_vf_30_phase_unwrap_seed_b0_v2": ["enable_b0"],
    "exp_vf_33_phase_navigator_lock_v2": ["enable_b0"],
    # B1 field (21/22/23/24 moved to the faithful multi_acquisition path — see FAITHFUL_ARMS)
    "exp_vf_27_reciprocity_b1minus_sim_v2": ["enable_b1"],
    # PSF / resolution / MTF (01 moved to faithful subvoxel_sr — see FAITHFUL_ARMS)
    "exp_vf_02_psf_deconvolution_v2": ["enable_gibbs"],
    "exp_vf_32_mtf_deapodization_v2": ["enable_gibbs"],
    # trajectory / gradient
    "exp_vf_03_trajectory_autocalib_v2": ["enable_eddy_current"],
    "exp_vf_31_kspace_phase_ramp_conjugator_v2": ["enable_motion"],
    "exp_vf_35_differentiable_trajectory_v2": ["enable_eddy_current"],
    # general marker-conditioned oracle
    "experiment_vf_method_a_cross_attention_v2": ["enable_motion", "enable_b0", "enable_b1"],
    "experiment_vf_method_b_hyper_mamba_v2": ["enable_motion", "enable_b0", "enable_b1"],
}

# Stated acceleration per undersampling arm (must match digital_twin.acceleration).
EXPECTED_ACCEL = {
    "exp_vf_05_csm_boundary_extrap_v2": 6.0,
    "exp_vf_08_deep_dc_forcing_v2": 8.0,
}


@pytest.mark.parametrize("stem", sorted(EXPECTED_AXIS))
def test_vf_arm_has_controlled_digital_twin(stem: str) -> None:
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    dt = cfg.physics.digital_twin
    assert dt.enabled is True, f"{stem}: digital_twin must be enabled"
    for flag in EXPECTED_AXIS[stem]:
        assert getattr(dt, flag) is True, f"{stem}: expected {flag} on (physics axis)"


@pytest.mark.parametrize("stem, accel", sorted(EXPECTED_ACCEL.items()))
def test_vf_undersampling_acceleration_matches_hypothesis(stem: str, accel: float) -> None:
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    dt = cfg.physics.digital_twin
    assert dt.enable_undersampling is True
    assert dt.acceleration == accel, f"{stem}: accel {dt.acceleration} != stated R={accel}"


def test_iterative_refinement_block_removed_from_subvoxel() -> None:
    # iterative_refinement has zero consumers in src/ — confirm it is gone.
    text = (VF_DIR / "exp_vf_01_subvoxel_superres_v2.yaml").read_text()
    assert "iterative_refinement:" not in text


# Arms whose described method needs an acquisition structure the single-frame
# VF pipeline cannot provide → honest provenance stamp (VF review 2026-06-02).
EXPECTED_FIDELITY = {
    "exp_vf_03_trajectory_autocalib_v2": "degradation_proxy",
    "exp_vf_04_b0_drift_demodulation_v2": "degradation_proxy",
    "exp_vf_10_bssfp_banding_mitigation_sim_v2": "degradation_proxy",
    # ff703c191 ("real-reference for the phase-B0 arms") migrated this arm to
    # multiecho_radial_b0_r2star and moved its status degradation_proxy ->
    # testable_on_real_b0, without updating this map -- which is what left the
    # test red on `dev` (#826). The value tracks what an arm CLAIMS; an arm that
    # legitimately improves its fidelity updates its entry rather than dropping
    # out, so the stamp itself stays asserted.
    "exp_vf_26_fid_navigator_b0_sim_v2": "testable_on_real_b0",
    "exp_vf_29_bssfp_banding_freq_b0_sim_v2": "degradation_proxy",
    "exp_vf_35_differentiable_trajectory_v2": "degradation_proxy",
}


def _metadata_extra(metadata: object, key: str) -> object | None:
    """Read a free-form metadata key off the typed schema or a raw dict.

    ``TrainingSettings.metadata`` was a bare ``dict`` until the typed
    ``ExperimentMetadataSchema`` was mounted on it (corpus review, T0.4);
    ``acquisition_fidelity`` is not a declared field, so it lives in the
    model's ``extra="allow"`` bag and neither ``[...]`` nor ``.get(...)``
    reaches it any more. One reader for both shapes, so this test says what it
    means whichever one it is handed.
    """
    if isinstance(metadata, dict):
        return metadata.get(key)
    extra = getattr(metadata, "model_extra", None) or {}
    return extra.get(key, getattr(metadata, key, None))


@pytest.mark.parametrize("stem, status", sorted(EXPECTED_FIDELITY.items()))
def test_overclaimed_arms_carry_honest_fidelity_stamp(stem: str, status: str) -> None:
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    af = _metadata_extra(cfg.metadata, "acquisition_fidelity")
    assert af is not None, f"{stem}: missing acquisition_fidelity provenance"
    assert af["status"] == status
    # the caveat must name both what the method needs and that it is not faithful
    assert af["method_requires"] and af["pipeline_provides"] and af["caveat"]


# Arms upgraded to faithful multi-acquisition synthesis (VF review 2026-06-03).
# stem -> (method, model in_channels, out_channels).
FAITHFUL_ARMS = {
    "exp_vf_21_dual_echo_phase_b0_sim_v2": ("dual_echo", 2, 2),
    "exp_vf_22_bloch_siegert_b1plus_sim_v2": ("bloch_siegert", 2, 2),
    "exp_vf_23_afi_b1plus_sim_v2": ("afi", 2, 2),
    "exp_vf_24_double_angle_b1plus_sim_v2": ("double_angle", 2, 2),
    "exp_vf_28_mrf_dictionary_b0b1_sim_v2": ("mrf", 16, 16),  # 16-timepoint transient
    # 8 LR frames + 2*8 shift-conditioning maps -> HR (2026-07-26 ladder)
    "exp_vf_01_subvoxel_superres_v2": ("subvoxel_sr", 24, 1),
    "exp_vf_01_subvoxel_superres_blind_v2": ("subvoxel_sr", 24, 1),
    "exp_vf_01_subvoxel_superres_oracle_v2": ("subvoxel_sr", 24, 1),
    "exp_vf_07_lowrank_temporal_subspace_v2": ("lowrank_temporal", 8, 8),  # 8-frame series
}


@pytest.mark.parametrize(
    "stem, method, in_ch, out_ch",
    [(s, m, i, o) for s, (m, i, o) in sorted(FAITHFUL_ARMS.items())],
)
def test_faithful_arms_wired_to_multi_acquisition(
    stem: str, method: str, in_ch: int, out_ch: int
) -> None:
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    assert cfg.training.training_mode == "multi_acquisition"
    ma = cfg.physics.multi_acquisition
    assert ma.enabled is True and ma.method == method
    assert cfg.model.in_channels == in_ch and cfg.model.out_channels == out_ch
    assert _metadata_extra(cfg.metadata, "acquisition_fidelity")["status"] == "faithful_simulated"


# ---------------------------------------------------------------------------
# Audit-failure regressions (cluster smoke 2026-06-03)
#
# Every VF arm failed the strict audit: 50 on a `validation_split_redundancy`
# WARNING (the dead `validation.split` knob — CLAUDE.md #15), 6 multi-acq arms
# on a `coil_processing_consistency` / `domain_alignment` ERROR (svd coil
# compression implied in_channels=2*nvc, but the model consumes a synthesised
# acquisition *stack* whose channel count is decoupled from the coil count),
# and exp_vf_01 crashed at train on a 3-D patch. These guard the fixes.
# ---------------------------------------------------------------------------
import yaml  # noqa: E402

from spectramr.infrastructure.validation.config_health_checker import (  # noqa: E402
    ConfigHealthChecker,
)
from tests.utils.corpus import tracked_yamls  # noqa: E402

ALL_VF_YAMLS = tracked_yamls(VF_DIR, recursive=False)


@pytest.mark.parametrize("path", ALL_VF_YAMLS, ids=lambda p: p.stem)
def test_no_vf_arm_carries_dead_validation_split(path: pathlib.Path) -> None:
    """`validation.split` has no reader in src/ (CLAUDE.md #15) — the manifest /
    `data.validation_split` drive the split, so the knob must not reappear."""
    raw = yaml.safe_load(path.read_text()) or {}
    assert (raw.get("validation") or {}).get("split") is None, (
        f"{path.stem}: dead 'validation.split' knob is back — remove it "
        "(use data.validation_split for a random split; manifest is authoritative)"
    )


@pytest.mark.parametrize("stem", sorted(FAITHFUL_ARMS), ids=lambda s: s)
def test_faithful_multi_acq_arms_pass_coil_channels(stem: str) -> None:
    """Synthesised-stack arms decouple in_channels from the coil count, so they
    must use passthrough coils (`none`); svd would re-impose in_channels==2*nvc."""
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    assert cfg.data.coils.processing_mode == "none"
    assert cfg.physics.coil_processing.compression.method == "none"


def test_subvoxel_arm_is_2d() -> None:
    """exp_vf_01 (subvoxel_sr) is a 2-D method; a 3-D patch produced a
    5-D batch that crashed base.train_step's 4-D unpack (cluster 2026-06-03)."""
    cfg = TrainingSettings.from_yaml(str(VF_DIR / "exp_vf_01_subvoxel_superres_v2.yaml"))
    # `data.patch_size` moved to `data.sampling.patch_size` on 2026-07-31
    # (RENAMES, posture `fold`): a YAML declaration still folds, but a PYTHON
    # read of the legacy spelling raises AttributeError -- which is how this
    # assertion, not the arm, went red (#826). The arm is still 2-D: the
    # canonical path reads (192, 192, 1).
    assert tuple(cfg.data.sampling.patch_size)[2] == 1


def test_subvoxel_arm_upsample_knob_is_spelled_for_its_backbone() -> None:
    """The subvoxel backbone must upsample by exactly ``sr_scale``.

    2026-07-26 swapped edsr -> restormer. edsr spells the knob ``scale_factor``;
    restormer spells it ``scale`` and DEFAULTS TO 4. A stale ``scale_factor: 2``
    is silently dropped by the factory signature filter, so the model would
    build at 4x and emit 512x512 against a 256x256 target. The same trap fires
    for swinir (``upscale``) and efficient_transformer (``upscale_factor``).
    """
    import inspect

    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    stem = "exp_vf_01_subvoxel_superres_v2"
    cfg = TrainingSettings.from_yaml(str(VF_DIR / f"{stem}.yaml"))
    entry = MODEL_REGISTRY.get(cfg.model.model_type)
    assert entry is not None, f"{cfg.model.model_type} is not a registered model"

    params = set(inspect.signature(entry["class"].__init__).parameters) - {"self"}
    kwargs = dict(cfg.model.model_kwargs)
    dropped = sorted(set(kwargs) - params)
    assert not dropped, (
        f"model_kwargs {dropped} are not parameters of "
        f"{cfg.model.model_type}.__init__ and will be silently dropped"
    )

    ma = cfg.physics.multi_acquisition
    # n frames + 2n shift-conditioning maps (2026-07-26 ladder). Identical
    # across blind/recovered/oracle so only the map CONTENT varies.
    assert cfg.model.in_channels == ma.n_frames * 3, (
        "input width must be n_frames frames plus 2*n_frames shift maps"
    )
    upsample_keys = {"scale", "scale_factor", "upscale", "upscale_factor"}
    knobs = [k for k in kwargs if k in upsample_keys]
    assert len(knobs) == 1, f"declare exactly one upsample knob, got {knobs}"
    assert kwargs[knobs[0]] == ma.sr_scale


@pytest.mark.parametrize("path", ALL_VF_YAMLS, ids=lambda p: p.stem)
def test_vf_arms_clear_audit_channel_and_split_checks(path: pathlib.Path) -> None:
    """Direct regression on the three checks that failed every VF arm."""
    cfg = TrainingSettings.from_yaml(str(path))
    chk = ConfigHealthChecker()

    def _norm(x: object) -> list:
        return x if isinstance(x, list) else ([] if x is None else [x])

    results = (
        _norm(chk.check_domain_alignment(cfg))
        + _norm(chk.check_coil_processing_consistency(cfg))
        + _norm(chk.check_validation_split_redundancy(cfg))
    )
    offenders = [
        (r.check_name, r.severity)
        for r in results
        if (not r.passed) or r.severity in ("warning", "error")
    ]
    assert not offenders, f"{path.stem}: audit checks still tripping: {offenders}"
