"""Training Strategy Factory - Schema-Driven Dispatcher

Resolves strategy classes based on config.training schema, replacing legacy
training_mode string-based dispatch with typed schema dispatch.

Design:
- Primary: config.training.strategy_class (explicit class path)
- Fallback: Infer from typed training schema (config.training.gan, config.training.diffusion, etc.)
- Deprecated: config.training_mode string (logged warning, remove in future release)
"""

import importlib
import logging
from typing import Any

from mriforge.domain.exceptions import ConfigurationError
from mriforge.models.capabilities import StrategyCapabilities

logger = logging.getLogger(__name__)

# Cache for out-of-tree strategy short-name -> dotted-path mappings, read once
# from the ``mriforge.strategies`` entry-point group.
_plugin_strategy_paths: dict[str, str] | None = None

# Dotted paths already announced at INFO by ``_load_strategy_class``. The import
# itself is cached by ``sys.modules``, so the second resolution is not a load and
# should not read like one.
_LOGGED_STRATEGY_PATHS: set[str] = set()


def _load_plugin_strategy_paths() -> dict[str, str]:
    """Out-of-tree strategy short-name → dotted-path map from entry-points.

    Strategies are a hardcoded dotted-path dict (NOT a decorator registry), so a
    plugin distribution declares ``[project.entry-points."mriforge.strategies"]``
    mapping a short name to the strategy's dotted path. Consulted ONLY on a miss
    against the static ``STRATEGY_CLASS_PATHS`` (which stays the SSOT), so an
    unknown name still raises (pitfall #9). A full dotted ``strategy_class`` FQN
    already resolves via the import path below and needs no entry-point.
    """
    global _plugin_strategy_paths
    if _plugin_strategy_paths is None:
        import importlib.metadata

        from mriforge.plugins import STRATEGY_ENTRY_POINT_GROUP

        paths: dict[str, str] = {}
        try:
            for ep in importlib.metadata.entry_points(group=STRATEGY_ENTRY_POINT_GROUP):
                paths[ep.name] = ep.value
        except Exception as exc:  # pragma: no cover - importlib edge cases
            logger.debug("strategy entry-point enumeration failed: %s", exc)
        _plugin_strategy_paths = paths
    return _plugin_strategy_paths


class TrainingStrategyFactory:
    """Factory for resolving training strategy classes from configuration.

    Dispatch priority:
    1. config.training.strategy_class (explicit)
    2. Typed training schema (config.training.gan, etc.)
    3. Legacy config.training_mode (deprecated, logs warning)
    """

    # Strategy class path registry (for convenience when using short names)
    STRATEGY_CLASS_PATHS = {
        # Core strategies
        "gan": "mriforge.infrastructure.training.strategies.gan.GANTrainingStrategy",
        "diffusion": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "reconstruction": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "vae": "mriforge.infrastructure.training.strategies.vae.VAETrainingStrategy",
        "vqvae": "mriforge.infrastructure.training.strategies.vae.VQVAETrainingStrategy",
        "mae": "mriforge.infrastructure.training.strategies.pretraining.MAEPretrainingStrategy",
        "ssl": "mriforge.infrastructure.training.strategies.pretraining.MAEPretrainingStrategy",
        "physics_driven": "mriforge.infrastructure.training.strategies.physics_driven_strategy.PhysicsDrivenTrainingStrategy",
        "pinn": "mriforge.infrastructure.training.strategies.pinn_strategy.ConcretePINNSensitivityStrategy",
        # Specialized strategies
        "domain_adaptation": "mriforge.infrastructure.training.strategies.domain_adaptation.DomainAdaptationTrainingStrategy",
        "volumetric": "mriforge.infrastructure.training.strategies.volumetric.TRELLISTrainingStrategy",
        "ttt": "mriforge.infrastructure.training.strategies.test_time_adaptation_strategy.TttAdaptationStrategy",
        "meta_learning": "mriforge.infrastructure.training.strategies.meta_learning_strategy.MetaLearningTrainingStrategy",
        "disentangled": "mriforge.infrastructure.training.strategies.disentangled_strategy.DisentangledTrainingStrategy",
        "cross_field_translation": "mriforge.infrastructure.training.strategies.cross_field_translation_strategy.CrossFieldTranslationStrategy",
        "field_cocycle": "mriforge.infrastructure.training.strategies.field_cocycle_strategy.FieldCocycleTranslationStrategy",
        "field_flow": "mriforge.infrastructure.training.strategies.field_flow_strategy.FieldFlowStrategy",
        "field_bridge": "mriforge.infrastructure.training.strategies.field_bridge_strategy.FieldBridgeStrategy",
        "ulf_map": "mriforge.infrastructure.training.strategies.ulf_map_strategy.UlfMapStrategy",
        "heteroscedastic_ulf": "mriforge.infrastructure.training.strategies.heteroscedastic_ulf_strategy.HeteroscedasticULFStrategy",
        "field_cold_diffusion": "mriforge.infrastructure.training.strategies.field_cold_diffusion_strategy.FieldColdDiffusionStrategy",
        "field_guided_diffusion": "mriforge.infrastructure.training.strategies.field_guided_diffusion_strategy.FieldGuidedDiffusionStrategy",
        "ulf_dps": "mriforge.infrastructure.training.strategies.ulf_dps_strategy.UlfDpsStrategy",
        "generative_refiner": "mriforge.infrastructure.training.strategies.generative_refiner_strategy.GenerativeRefinerStrategy",
        "field_conditioned_inr": "mriforge.infrastructure.training.strategies.field_conditioned_inr_strategy.FieldConditionedINRStrategy",
        "monotone_field": "mriforge.infrastructure.training.strategies.monotone_field_strategy.MonotoneFieldStrategy",
        "ulf_redegrad_tta": "mriforge.infrastructure.training.strategies.ulf_redegrad_tta_strategy.UlfReDegradationTTAStrategy",
        "quality_matching": "mriforge.infrastructure.training.strategies.quality_matching_strategy.QualityMatchingStrategy",
        "field_fno": "mriforge.infrastructure.training.strategies.field_fno_strategy.FieldFNOStrategy",
        "bloch_field": "mriforge.infrastructure.training.strategies.bloch_field_strategy.BlochFieldStrategy",
        "bloch_synth": "mriforge.infrastructure.training.strategies.bloch_synth_strategy.BlochSynthesisStrategy",
        "steerable_synthesis": "mriforge.infrastructure.training.strategies.steerable_synthesis_strategy.SteerableSynthesisStrategy",
        "doob_bridge": "mriforge.infrastructure.training.strategies.doob_bridge_strategy.DoobBridgeStrategy",
        "confluence": "mriforge.infrastructure.training.strategies.confluence_strategy.ConfluenceStrategy",
        "brenier_synthesis": "mriforge.infrastructure.training.strategies.brenier_synthesis_strategy.BrenierSynthesisStrategy",
        "mccann_field_path": "mriforge.infrastructure.training.strategies.mccann_field_path_strategy.McCannFieldPathStrategy",
        "fisher_rao_geodesic": "mriforge.infrastructure.training.strategies.fisher_rao_geodesic_strategy.FisherRaoGeodesicStrategy",
        "lora_modulation": "mriforge.infrastructure.training.strategies.lora_modulation_strategy.LoRAModulationStrategy",
        "koopman_field": "mriforge.infrastructure.training.strategies.koopman_field_strategy.KoopmanFieldStrategy",
        "scattering_besov": "mriforge.infrastructure.training.strategies.scattering_besov_strategy.ScatteringBesovStrategy",
        "recoverability_vib": "mriforge.infrastructure.training.strategies.recoverability_vib_strategy.RecoverabilityVIBStrategy",
        "cartoon_texture_safe": "mriforge.infrastructure.training.strategies.cartoon_texture_safe_strategy.CartoonTextureSafeStrategy",
        "field_wiener": "mriforge.infrastructure.training.strategies.field_wiener_strategy.FieldWienerStrategy",
        "disentangled_vae": "mriforge.infrastructure.training.strategies.disentangled_vae_strategy.DisentangledVAETrainingStrategy",
        "masked": "mriforge.infrastructure.training.strategies.masked_strategy.MaskedPretrainingStrategy",
        "padnet": "mriforge.infrastructure.training.strategies.padnet_strategy.PaDNetTrainingStrategy",
        "swarm": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "graph_cold_diffusion": "mriforge.infrastructure.training.strategies.graph_cold_diffusion_strategy.GraphColdDiffusionStrategy",
        "b0_mapping": "mriforge.infrastructure.training.strategies.b0_mapping_strategy.B0MappingStrategy",
        # [Proposal 1] Lie-algebraic BCH effective-generator identification of
        # the composite degradation operator. See docs/strategies_reference.rst.
        "operator_id_bch": "mriforge.infrastructure.training.strategies.operator_id_bch_strategy.OperatorIdBCHTrainingStrategy",
        "operator_id": "mriforge.infrastructure.training.strategies.operator_id_bch_strategy.OperatorIdBCHTrainingStrategy",
        # Contrast/field-agnostic bundle (2026-06-29 design): M3 LCAH — the
        # `lcah_encoder` model declares training_mode="acq_hypernetwork", so
        # this key is what makes that registration resolvable.
        "acq_hypernetwork": "mriforge.infrastructure.training.strategies.hypernetwork_strategy.AcquisitionHypernetworkStrategy",
        # M4 DL-BAE — dispersion-latent Bloch autoencoder.
        "dispersion_bloch_ae": "mriforge.infrastructure.training.strategies.dispersion_bloch_ae_strategy.DispersionBlochAEStrategy",
        "rectified_flow": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "dual_task": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "flow_matching": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        # Phase-3 (2026-05-22) re-implemented deleted models:
        # `generative` drives maximum-likelihood training for the
        # normalising-flow models (glow, equivariant_flow);
        # `progressive_gan` extends the GAN loop with a phase scheduler.
        "generative": "mriforge.infrastructure.training.strategies.generative.GenerativeTrainingStrategy",
        "progressive_gan": "mriforge.infrastructure.training.strategies.progressive_gan_strategy.ProgressiveGANStrategy",
        "beta_vae_gan": "mriforge.infrastructure.training.strategies.betavaegan_strategy.BetaVAEGANStrategy",
        # MRIxFields2026 baseline: unpaired two-generator CycleGAN. The batch
        # ``target`` is a real B-domain sample for disc_b, NEVER a pixel-L1 target
        # (see cyclegan_strategy.py; CLAUDE.md pitfall #16).
        "cyclegan": "mriforge.infrastructure.training.strategies.cyclegan_strategy.CycleGANTrainingStrategy",
        "cycle_gan": "mriforge.infrastructure.training.strategies.cyclegan_strategy.CycleGANTrainingStrategy",
        # MRIxFields2026 baseline: single-generator Contrastive Unpaired
        # Translation (CUT). One generator + one PatchGAN + PatchNCE — no cycle,
        # no pixel-L1 to ``target`` (which is only a real B-domain sample for D;
        # see cut_strategy.py; CLAUDE.md pitfall #16).
        "cut": "mriforge.infrastructure.training.strategies.cut_strategy.CUTTrainingStrategy",
        # MRIxFields2026 baseline: multi-domain (five FIELD levels) StarGAN v2.
        # Four nets — G(x,s) + mapping F(z,y) + style-encoder E(x,y) trained on
        # opt_g, discriminator D(x,y) on opt_d. Adversarial (gan_lsgan) + style-recon
        # + style-diversification + cycle; D adds an r1 penalty. No paired pixel-L1
        # to target (see stargan_v2_strategy.py; CLAUDE.md pitfall #16).
        "stargan_v2": "mriforge.infrastructure.training.strategies.stargan_v2_strategy.StarGANv2TrainingStrategy",
        # Legacy/specialized strategies (now with BaseTrainingStrategy)
        "cycle_bloch": "mriforge.infrastructure.training.strategies.cycle_bloch_strategy.CycleBlochStrategy",
        "noise_to_noise": "mriforge.infrastructure.training.strategies.n2n_strategy.NoiseToNoiseStrategy",
        "n2n": "mriforge.infrastructure.training.strategies.n2n_strategy.NoiseToNoiseStrategy",
        "self_supervised_reconstruction": "mriforge.infrastructure.training.strategies.ssdu_strategy.SSDUReconstructionStrategy",
        "ssdu": "mriforge.infrastructure.training.strategies.ssdu_strategy.SSDUReconstructionStrategy",
        # Robust SSDU (Noisier2Noise, Millard & Chiew 2024): same class; the
        # noisier2noise_correction knob (required true for this key) toggles the
        # synthetic-noise augmentation.
        "robust_ssdu": "mriforge.infrastructure.training.strategies.ssdu_strategy.SSDUReconstructionStrategy",
        # Equivariant Imaging (Chen et al. 2021) + Robust EI (2022). Both keys
        # resolve to the same class; 'robust_ei' requires robust_correction=true.
        "equivariant_imaging": "mriforge.infrastructure.training.strategies.equivariant_imaging_strategy.EquivariantImagingStrategy",
        "robust_ei": "mriforge.infrastructure.training.strategies.equivariant_imaging_strategy.EquivariantImagingStrategy",
        # Ambient Diffusion / A-DPS (Daras et al. 2023; Aali et al.): SSDU Λ/Θ
        # split lifted onto a diffusion prior; A-DPS inference reuses DDS.
        "ambient_diffusion": "mriforge.infrastructure.training.strategies.ambient_diffusion_strategy.AmbientDiffusionStrategy",
        "diff_siren": "mriforge.infrastructure.training.strategies.diff_siren.DIFFSirenStrategy",
        # Virtual Fiducial motion-correction strategies
        "virtual_fiducial": "mriforge.infrastructure.training.strategies.virtual_fiducial_strategy.ConcreteVirtualFiducialStrategy",
        "distillation": "mriforge.infrastructure.training.strategies.distillation_strategy.ConcreteDistillationStrategy",
        "motion_meta": "mriforge.infrastructure.training.strategies.motion_meta_strategy.ConcreteMotionMetaTrainingStrategy",
        "tto": "mriforge.infrastructure.training.strategies.tto_strategy.ConcreteTTOStrategy",
        "test_time_optimization": "mriforge.infrastructure.training.strategies.tto_strategy.ConcreteTTOStrategy",
        # SPECTRA test-time adaptation (plan §D): freeze backbone, adapt norm/adapter
        # params per subject via self-supervised k-space consistency. The single
        # training-flavoured SPECTRA workstream.
        "spectra_tta": "mriforge.infrastructure.training.strategies.spectra_tta_strategy.SpectraTestTimeAdaptationStrategy",
        "vf_admm": "mriforge.infrastructure.training.strategies.vf_admm_strategy.ConcreteVFADMMStrategy",
        "multi_acquisition": "mriforge.infrastructure.training.strategies.multi_acquisition_strategy.ConcreteMultiAcquisitionStrategy",
        "trajectory_recon": "mriforge.infrastructure.training.strategies.trajectory_recon_strategy.TrajectoryReconstructionStrategy",
        "pma_varnet": "mriforge.infrastructure.training.strategies.pma_varnet_strategy.ConcretePMAVarNetStrategy",
        "multi": "mriforge.infrastructure.training.strategies.pipeline_strategy.MultiTrainingStrategy",
        # Aliases for training_mode values used in experiment YAMLs
        "cold_diffusion": "mriforge.infrastructure.training.strategies.graph_cold_diffusion_strategy.GraphColdDiffusionStrategy",
        "kspace_cold_diffusion": "mriforge.infrastructure.training.strategies.graph_cold_diffusion_strategy.GraphColdDiffusionStrategy",
        "hybrid": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "mesh": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "multi_task": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "swarm_learning": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "test_time_adaptation": "mriforge.infrastructure.training.strategies.tto_strategy.ConcreteTTOStrategy",
        "3d_generation": "mriforge.infrastructure.training.strategies.volumetric.TRELLISTrainingStrategy",
        "federated": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "adaptation": "mriforge.infrastructure.training.strategies.domain_adaptation.DomainAdaptationTrainingStrategy",
        "disentangled_diffusion": "mriforge.infrastructure.training.strategies.disentangled_diffusion_strategy.DisentangledDiffusionStrategy",
        # X-Diffusion (cross-modal / multi-contrast). The XDiffusionTrainingStrategy
        # advertises ``expected_modes=("diffusion", "3d_generation",
        # "cross_modal_diffusion")``; before these entries the dispatcher had no
        # short-name route to it and YAMLs with ``model_type: x_diffusion`` silently
        # fell back to plain DiffusionTrainingStrategy (CLAUDE.md pitfall #9 / #10).
        "x_diffusion": "mriforge.infrastructure.training.strategies.diffusion.XDiffusionTrainingStrategy",
        "cross_modal_diffusion": "mriforge.infrastructure.training.strategies.diffusion.XDiffusionTrainingStrategy",
        "guided_sr": "mriforge.infrastructure.training.strategies.guided_sr_strategy.GuidedSuperResolutionStrategy",
        # [Breakthrough 2026] SLE-driven compressed-sensing acquisition.
        # The strategy is a thin alias of ReconstructionTrainingStrategy; the
        # SLE-specific behaviour comes from the YAML's `data.acceleration.mask_type:
        # sle_kappa` field (dispatched through `MaskType.SLE_KAPPA`). See
        # src/data/transforms/sle_trajectory.py and
        # docs/breakthrough_methods.rst (SLE Acquisition Trajectories).
        "sle_compressed_sensing": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        # [Breakthrough 2026] Connes spectral-triple ULF→HF translation. The
        # operator-algebraic intertwining penalty is provided by the
        # `spectral_triple_intertwining` loss (registered via
        # @register_loss in src/models/losses/spectral_triple_loss.py); the
        # strategy itself reuses ReconstructionTrainingStrategy.
        "spectral_triple": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        # [Breakthrough 2026 — Batches 1 + 2] Thin reconstruction / diffusion
        # aliases. The paradigm-specific behaviour comes from each YAML's
        # data + loss + metric block; the math primitives live under
        # src/infrastructure/physics/, src/models/blocks/, and
        # src/models/diffusion/. See docs/breakthrough_methods.rst.
        "tropical_mrf": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "fisher_rao_flow": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "sheaf_kspace": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "resetting_diffusion": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "hyperbolic_cardiac": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "heisenberg_phase": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "hjb_trajectory": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "primary_ideal": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "symplectic_bloch": "mriforge.infrastructure.training.strategies.cycle_bloch_strategy.CycleBlochStrategy",
        "frontdoor_federated": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "hodge_motion": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "levy_diffusion": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "koopman_fmri": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "sheaf_multi_contrast": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "hjb_waveform": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "stein_federated": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "tropical_quantitative_maps": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        "frontdoor_scanner": "mriforge.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy",
        # GeoMamba-ULF: foreground-restricted contrast-conditioned Mamba
        # SR for ULF→HF translation. See docs/geomamba_ulf.rst.
        "geomamba_ulf": "mriforge.infrastructure.training.strategies.geomamba_ulf_strategy.GeoMambaULFStrategy",
        # IMPLEMENTATION_SPEC.md §8 (Phase 2) — Hilbert State-Space k-space
        # Completion: Hilbert + BiMamba + Hermitian projection + DC clamp.
        "hssc": "mriforge.infrastructure.training.strategies.hssc_strategy.HSSCStrategy",
        # IMPLEMENTATION_SPEC.md §10 (Phase 4) — Dual-Traversal Noise2Self
        # self-supervised denoising with J-invariance mask.
        "dtn2s": "mriforge.infrastructure.training.strategies.dtn2s_strategy.DTN2SStrategy",
        # synthetic_marker_research_plan.md §4.3 — privileged-marker
        # teacher-student training with marker-density curriculum +
        # gradient-reversal domain discriminator.
        "privileged_learning": "mriforge.infrastructure.training.strategies.privileged_learning_strategy.PrivilegedLearningStrategy",
        "privileged": "mriforge.infrastructure.training.strategies.privileged_learning_strategy.PrivilegedLearningStrategy",
        # IMPLEMENTATION_SPEC.md §2.7 — slice-to-volume isotropic
        # synthesis with through-plane + orthogonal-reformat consistency.
        "slice_to_volume": "mriforge.infrastructure.training.strategies.slice_to_volume_strategy.SliceToVolumeStrategy",
        "2d_to_3d": "mriforge.infrastructure.training.strategies.slice_to_volume_strategy.SliceToVolumeStrategy",
        # ============================================================
        # v6.1 — paradigm-expansion strategies (PR-3..PR-25 cross-cuts)
        # ============================================================
        "calibration": "mriforge.infrastructure.training.strategies.conformal_calibration_strategy.ConformalCalibrationStrategy",
        "validation_badge": "mriforge.infrastructure.training.strategies.validation_badge_strategy.ValidationBadgeStrategy",
        "learnable_acquisition": "mriforge.infrastructure.training.strategies.loupe_strategy.LOUPEStrategy",
        "loupe": "mriforge.infrastructure.training.strategies.loupe_strategy.LOUPEStrategy",
        "pilot": "mriforge.infrastructure.training.strategies.pilot_strategy.PILOTStrategy",
        "active_acquisition": "mriforge.infrastructure.training.strategies.bald_acquisition_strategy.BALDAcquisitionStrategy",
        "bald": "mriforge.infrastructure.training.strategies.bald_acquisition_strategy.BALDAcquisitionStrategy",
        "multi_param_mapping": "mriforge.infrastructure.training.strategies.multi_param_mapping_strategy.OneShotMultiParameterStrategy",
        "multi_contrast_contrastive": "mriforge.infrastructure.training.strategies.multi_contrast_contrastive_strategy.MultiContrastContrastiveStrategy",
        "edm": "mriforge.infrastructure.training.strategies.edm_training_strategy.EDMTrainingStrategy",
        "pnp": "mriforge.infrastructure.training.strategies.pnp_strategy.PnPStrategy",
        "pnp_red": "mriforge.infrastructure.training.strategies.pnp_strategy.PnPStrategy",
        "concomitant_aware_recon": "mriforge.infrastructure.training.strategies.concomitant_aware_recon_strategy.ConcomitantAwareReconStrategy",
        "caur": "mriforge.infrastructure.training.strategies.concomitant_aware_recon_strategy.ConcomitantAwareReconStrategy",
        "xfield_fm": "mriforge.infrastructure.training.strategies.xfield_fm_strategy.XFieldFMStrategy",
        "scas": "mriforge.infrastructure.training.strategies.scas_strategy.SCASStrategy",
        "federated_dp_conformal": "mriforge.infrastructure.training.strategies.federated_dp_conformal_strategy.FederatedDPConformalULFStrategy",
        # ============================================================
        # ULF Phase-1 strategies (ULF-PR-4, 7, 9, 13)
        # ============================================================
        "kspace_inr": "mriforge.infrastructure.training.strategies.kspace_inr_strategy.KSpaceINRStrategy",
        "bloch_consistent_denoising": "mriforge.infrastructure.training.strategies.bloch_consistent_denoising_strategy.BlochConsistentDenoisingStrategy",
        # Physics-in-the-loop B0 field fit (audit 2026-07 I1) — the REAL B0
        # estimator (map_b0 SSOT, graded in Hz). b0_mapping below is deformable
        # registration, NOT a field fit.
        "multi_echo_b0_fit": "mriforge.infrastructure.training.strategies.multi_echo_b0_fit_strategy.MultiEchoB0FitStrategy",
        "girf_aware": "mriforge.infrastructure.training.strategies.girf_aware_strategy.GIRFAwareStrategy",
        "inverse_bloch_phase": "mriforge.infrastructure.training.strategies.inverse_bloch_phase_strategy.InverseBlochPhaseStrategy",
        # ============================================================
        # M-tier + L-tier follow-ups (PR-12, PR-13, PR-17, PR-18)
        # ============================================================
        "low_rank_sparse": "mriforge.infrastructure.training.strategies.low_rank_sparse_strategy.LowRankSparseStrategy",
        # Regime verticals (imaging-regime x task contract). The maturity ledger
        # walks THIS dict, so a strategy absent here is invisible to it however
        # well it is tagged.
        "phase_contrast_flow": "mriforge.infrastructure.training.strategies.phase_contrast_flow_strategy.PhaseContrastFlowStrategy",
        "perfusion_kinetic": "mriforge.infrastructure.training.strategies.perfusion_kinetic_strategy.PerfusionKineticMappingStrategy",
        "mrs_quantification": "mriforge.infrastructure.training.strategies.mrs_quantification_strategy.MRSQuantificationStrategy",
        "qsm_pipeline": "mriforge.infrastructure.training.strategies.qsm_pipeline_strategy.QSMPipelineStrategy",
        "qspace_diffusion": "mriforge.infrastructure.training.strategies.qspace_diffusion_strategy.QSpaceDiffusionStrategy",
        "adversarial_robustness": "mriforge.infrastructure.training.strategies.adversarial_robustness_strategy.AdversarialRobustnessStrategy",
        "certified_robustness": "mriforge.infrastructure.training.strategies.certified_robustness_strategy.CertifiedRobustnessStrategy",
        "sparse_frame": "mriforge.infrastructure.training.strategies.sparse_frame_strategy.SparseFrameStrategy",
        # ============================================================
        # Part-I A..F (PR-19, PR-21, PR-24 cover the new training loops)
        # ============================================================
        "stochastic_interpolants": "mriforge.infrastructure.training.strategies.stochastic_interpolants_strategy.StochasticInterpolantsStrategy",
        "jepa": "mriforge.infrastructure.training.strategies.jepa_strategy.JEPAStrategy",
        # ============================================================
        # ULF Phase-1 remainder (ULF-PR-5, 6, 8, 10, 11, 12)
        # ============================================================
        "mri_slam": "mriforge.infrastructure.training.strategies.mri_slam_strategy.MRISLAMStrategy",
        "noise_adaptive_score": "mriforge.infrastructure.training.strategies.noise_adaptive_score_strategy.NoiseAdaptiveScoreStrategy",
        "diffeomorphic_recon": "mriforge.infrastructure.training.strategies.diffeomorphic_recon_strategy.DiffeomorphicReconStrategy",
        "field_probe_coupled": "mriforge.infrastructure.training.strategies.field_probe_coupled_strategy.FieldProbeCoupledStrategy",
        "spin_sde": "mriforge.infrastructure.training.strategies.spin_sde_strategy.SpinSDEStrategy",
        "coord_kspace_gen": "mriforge.infrastructure.training.strategies.coord_kspace_gen_strategy.CoordKSpaceGenStrategy",
        # ============================================================
        # Integration plan ideas 3, 4, 6, 7 (idea 9 already in inverse_bloch_phase)
        # ============================================================
        "bloch_bottleneck": "mriforge.infrastructure.training.strategies.bloch_bottleneck_strategy.BlochBottleneckStrategy",
        "cycle_bloch_digital_twin": "mriforge.infrastructure.training.strategies.cycle_bloch_digital_twin_strategy.CycleBlochDigitalTwinStrategy",
        "cross_contrast_kspace_diffusion": "mriforge.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy.CrossContrastKspaceDiffusionStrategy",
        "synthetic_pathology_aug": "mriforge.infrastructure.training.strategies.synthetic_pathology_aug_strategy.SyntheticPathologyAugStrategy",
        # ============================================================
        # TODO/audit/audit_plan_novel.md — seven novel research ideas.
        # Phase 2/3/4 strategies (Ideas 6, 4, 2, 3, 5). Ideas 1 & 7 are
        # additive (model registry / audit hook) and need no strategy
        # entry of their own.
        # ============================================================
        "physics_equivariant_ssl": "mriforge.infrastructure.training.strategies.physics_equivariant_ssl_strategy.PhysicsEquivariantSSLStrategy",
        "phys_eq_ssl": "mriforge.infrastructure.training.strategies.physics_equivariant_ssl_strategy.PhysicsEquivariantSSLStrategy",
        "ib_active_acquisition": "mriforge.infrastructure.training.strategies.ib_active_acquisition_strategy.IBActiveAcquisitionStrategy",
        "ib_bald": "mriforge.infrastructure.training.strategies.ib_active_acquisition_strategy.IBActiveAcquisitionStrategy",
        "score_field_tomography": "mriforge.infrastructure.training.strategies.score_field_tomography_strategy.ScoreFieldTomographyStrategy",
        "bloch_equivariant_translation": "mriforge.infrastructure.training.strategies.bloch_equivariant_translation_strategy.BlochEquivariantTranslationStrategy",
        "riemannian_bloch_diffusion": "mriforge.infrastructure.training.strategies.riemannian_bloch_diffusion_strategy.RiemannianBlochDiffusionStrategy",
        # SFC / conformal §§1, 2, 4, 5 (TODO/audit_plan_novel.md lines 194-344).
        "adaptive_sfc_hssc": "mriforge.infrastructure.training.strategies.sfc_conformal_kspace_strategies.AdaptiveSFCHSSCStrategy",
        "conformal_diffusion_recon": "mriforge.infrastructure.training.strategies.sfc_conformal_kspace_strategies.ConformalDiffusionReconStrategy",
        "beltrami_motion_correction": "mriforge.infrastructure.training.strategies.beltrami_motion_cortical_strategies.BeltramiMotionCorrectionStrategy",
        "cortical_conformal_recon": "mriforge.infrastructure.training.strategies.beltrami_motion_cortical_strategies.CorticalConformalReconStrategy",
        "teichmuller_cold_diffusion": "mriforge.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy.TeichmullerColdDiffusionStrategy",
        # fMRI §§1-5 (TODO/audit_plan_novel_fmri.md fMRI section).
        "spatiotemporal_adaptive_sfc_recon": "mriforge.infrastructure.training.strategies.fmri_kspace_strategies.SpatiotemporalAdaptiveSFCReconStrategy",
        "beltrami_epi_distortion": "mriforge.infrastructure.training.strategies.fmri_kspace_strategies.BeltramiEPIDistortionStrategy",
        "cortical_conformal_fmri_recon": "mriforge.infrastructure.training.strategies.fmri_surface_strategies.CorticalConformalFMRIReconStrategy",
        "riemannian_dfc_diffusion": "mriforge.infrastructure.training.strategies.fmri_surface_strategies.RiemannianDFCDiffusionStrategy",
        "hrf_manifold_diffusion": "mriforge.infrastructure.training.strategies.fmri_surface_strategies.HRFManifoldDiffusionStrategy",
        # MRF §§1-5 (TODO/audit_plan_novel_fmri.md MRF section).
        "spatiotemporal_mrf_recon": "mriforge.infrastructure.training.strategies.mrf_kspace_strategies.SpatiotemporalMRFReconStrategy",
        "riemannian_mrf_diffusion": "mriforge.infrastructure.training.strategies.mrf_kspace_strategies.RiemannianMRFDiffusionStrategy",
        "conformal_mrf_dictless_recon": "mriforge.infrastructure.training.strategies.mrf_acquisition_strategies.ConformalMRFDictlessReconStrategy",
        "crlb_mrf_pulse_design": "mriforge.infrastructure.training.strategies.mrf_acquisition_strategies.CRLBMRFPulseDesignStrategy",
        "cross_scanner_mrf_harmonisation": "mriforge.infrastructure.training.strategies.mrf_acquisition_strategies.CrossScannerMRFHarmonisationStrategy",
        # ============================================================
        # PMPS — Physics-Mediated Paired Synthesis (2026-05-19).
        # Phase 1: tissue-parameter-prior pretraining. Thin alias to
        # DiffusionTrainingStrategy; paradigm-specific surface lives in
        # the new ``tissue_parameter_diffusion`` model + two physical-
        # constraint losses (see src/models/generators/tissue_diffusion/
        # and src/models/losses/physics/tissue_parameter_physics.py).
        # Phase 2: corruption-operator calibration. Concrete strategy
        # in corruption_calibration_strategy.py.
        # Phase 3: paired synthesis. Concrete strategy in
        # paired_synthesis_strategy.py.
        # Phase 4: data-efficiency harness. Concrete strategy in
        # data_efficiency_harness_strategy.py.
        # ============================================================
        "ib_vf": "mriforge.infrastructure.training.strategies.ib_vf_strategy.IBVFTrainingStrategy",
        "twin_dps": "mriforge.infrastructure.training.strategies.twin_dps_strategy.TwinLikelihoodDPSStrategy",
        "tissue_diffusion_pretrain": "mriforge.infrastructure.training.strategies.diffusion.DiffusionTrainingStrategy",
        "corruption_calibration": "mriforge.infrastructure.training.strategies.pmps_strategies.CorruptionCalibrationStrategy",
        "paired_synthesis": "mriforge.infrastructure.training.strategies.pmps_strategies.PairedSynthesisStrategy",
        "data_efficiency_harness": "mriforge.infrastructure.training.strategies.pmps_strategies.DataEfficiencyHarnessStrategy",
        # VF campaign Phase 2 breakthrough methods (B-1..B-4).
        "equivariance_conformal": "mriforge.infrastructure.training.strategies.equivariance_conformal_strategy.EquivarianceConformalCalibrationStrategy",
        "phys_residual_conformal": "mriforge.infrastructure.training.strategies.phys_residual_conformal_strategy.PhysResidualConformalStrategy",
        "bloch_manifold_dps": "mriforge.infrastructure.training.strategies.bloch_manifold_dps_strategy.BlochManifoldDPSStrategy",
        "se3_equivariant_navigator": "mriforge.infrastructure.training.strategies.se3_equivariant_navigator_strategy.SE3EquivariantNavigatorStrategy",
        "hamiltonian_acquisition": "mriforge.infrastructure.training.strategies.hamiltonian_acquisition_strategy.HamiltonianAcquisitionStrategy",
        # ============================================================
        # Frontier gap audit PR-4/5/7/9 (2026-05-29) — concrete strategy
        # classes (NOT generic-base aliases). Real paradigm math in
        # _compute_losses_impl. See TODO/backlog_frontier_gap_audit_2026_05.md.
        # ============================================================
        "bloch_schrodinger_bridge": "mriforge.infrastructure.training.strategies.schrodinger_bridge_strategy.BlochSchrodingerBridgeStrategy",
        "i2sb": "mriforge.infrastructure.training.strategies.schrodinger_bridge_strategy.BlochSchrodingerBridgeStrategy",
        "vf_consistency_distillation": "mriforge.infrastructure.training.strategies.vf_consistency_distillation_strategy.VFConsistencyDistillationStrategy",
        "universal_reconstruction": "mriforge.infrastructure.training.strategies.universal_reconstruction_strategy.UniversalReconstructionStrategy",
        "flow_matching_pfode": "mriforge.infrastructure.training.strategies.flow_matching_strategy.FlowMatchingStrategy",
    }

    def get_strategy_class(self, config: Any) -> type:
        """Resolve strategy class from config.

        Two rungs, most-explicit first:

        1. ``training.strategy_class`` -- a dotted path or a registered short name.
        2. ``training.training_mode`` -- a key in :attr:`STRATEGY_CLASS_PATHS`.

        A third rung used to sit *between* these: ``_infer_from_schema`` guessed
        ``DiffusionTrainingStrategy`` from a truthy ``training.timesteps`` /
        ``training.diffusion_type``. It outranked the user's explicit
        ``training_mode``, which is pitfall #9 (a silent fallback overriding a
        declaration). It was also unreachable in practice -- ``reject_flat_keys``
        on ``TrainingStrategyConfigSchema`` rejects both flat keys outright -- so
        it only ever fired for duck-typed mocks, and three test files had to null
        those attributes out to test this method at all. Removed 2026-07-19.

        Args:
            config: Training configuration with .training schema

        Returns:
            Strategy class to instantiate

        Raises:
            ConfigurationError: If strategy cannot be resolved
        """
        training = getattr(config, "training", None)
        if not training:
            raise ConfigurationError(
                "No 'training' section in configuration -- cannot resolve a training strategy."
            )

        # Priority 1: explicit strategy_class always wins.
        strategy_class_path = training.strategy_class
        if strategy_class_path:
            return self._load_strategy_class(strategy_class_path)

        # Priority 2: training_mode dispatch (O(1) registry lookup).
        training_mode = getattr(training, "training_mode", None)
        if training_mode:
            if training_mode in self.STRATEGY_CLASS_PATHS:
                logger.info(
                    f"Resolved strategy from training_mode='{training_mode}'. "
                    "Consider adding 'training.strategy_class' for explicit dispatch."
                )
                return self._load_strategy_class(training_mode)

            # Declared but unresolvable -- name it. The old message claimed "No
            # strategy specified" even when one plainly was.
            raise ConfigurationError(
                f"training.training_mode={training_mode!r} names no registered "
                "strategy. Add it to TrainingStrategyFactory.STRATEGY_CLASS_PATHS, "
                "or set 'training.strategy_class' to a dotted class path.\n"
                f"Valid strategies: {', '.join(sorted(self.STRATEGY_CLASS_PATHS))}"
            )

        raise ConfigurationError(
            "No strategy specified in v6.0 configuration. "
            "Add 'training.strategy_class' to your config. "
            "Example: training.strategy_class: 'mriforge.infrastructure.training.strategies.gan.GANTrainingStrategy'\n"
            f"Valid strategies: {', '.join(sorted(self.STRATEGY_CLASS_PATHS))}"
        )

    def _load_strategy_class(self, class_path: str) -> type:
        """Load strategy class from full module path or short name.

        Args:
            class_path: Full path like "mriforge.infrastructure.training.strategies.X.StrategyClass"
                       or short name like "gan", "diffusion"

        Returns:
            Strategy class
        """
        # Check if it's a short name (static SSOT first, then out-of-tree plugins)
        if class_path in self.STRATEGY_CLASS_PATHS:
            class_path = self.STRATEGY_CLASS_PATHS[class_path]
        else:
            plugin_strategies = _load_plugin_strategy_paths()
            if class_path in plugin_strategies:
                class_path = plugin_strategies[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            # Report the LOAD, not the lookup. The import is cached by
            # ``sys.modules`` but this line is not, and several independent
            # callers resolve the strategy per process (the health checker's
            # conditioning check, ``context_resolver``, ``ModelInitializer``) —
            # so an unconditional info printed "Loaded strategy: X" two or three
            # times per job and read as a repeated import.
            if class_path not in _LOGGED_STRATEGY_PATHS:
                _LOGGED_STRATEGY_PATHS.add(class_path)
                logger.info(f"Loaded strategy: {class_name}")
            else:
                logger.debug(f"Resolved strategy (already loaded): {class_name}")
            return strategy_class
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Failed to load strategy class '{class_path}': {e}. "
                f"Ensure the class exists and is importable."
            ) from e

    def get_strategy_capabilities(self, strategy: type | str) -> StrategyCapabilities:
        """Resolve the :class:`StrategyCapabilities` contract for a strategy.

        The flat ``STRATEGY_CLASS_PATHS`` dict carries only import strings (and
        ~25 short names alias the *same* class), so the contract rides on the
        resolved class as a ``ClassVar`` rather than per dict-entry -- this
        collapses the alias duplication (cached-cascade WS-X).

        Args:
            strategy: a strategy class, a short name ("gan", "diffusion", ...),
                or a fully-qualified dotted class path.

        Returns:
            The class's declared ``capabilities``, or the empty default
            ``StrategyCapabilities()`` ("unannotated, skip the check") when the
            class predates the contract.
        """
        strategy_class = (
            strategy if isinstance(strategy, type) else self._load_strategy_class(strategy)
        )
        caps = getattr(strategy_class, "capabilities", None)
        return caps if isinstance(caps, StrategyCapabilities) else StrategyCapabilities()

    def create_strategy(
        self,
        env: Any,
        logging_service: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Create and instantiate a strategy based on configuration.

        Args:
            env: Training environment containing config
            logging_service: Optional logging service
            **kwargs: Additional services (metrics_service, checkpoint_service, etc.)

        Returns:
            Instantiated strategy object
        """
        strategy_class = self.get_strategy_class(env.config)
        return strategy_class(env=env, logging_service=logging_service, **kwargs)


__all__ = ["TrainingStrategyFactory"]
