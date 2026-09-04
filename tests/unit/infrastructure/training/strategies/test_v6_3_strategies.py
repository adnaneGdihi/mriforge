"""v6.3 strategy registration smoke tests.

Plan: ULF Phase-1 remainder + Phase-2 pipelines + Integration plan idea
3, 4, 6, 7 follow-ups.
"""

from __future__ import annotations

import importlib

import pytest

from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory


V6_3_STRATEGIES: list[tuple[str, str, str]] = [
    # ULF Phase-1 remainder
    ("mri_slam",
     "spectramr.infrastructure.training.strategies.mri_slam_strategy",
     "MRISLAMStrategy"),
    ("noise_adaptive_score",
     "spectramr.infrastructure.training.strategies.noise_adaptive_score_strategy",
     "NoiseAdaptiveScoreStrategy"),
    ("diffeomorphic_recon",
     "spectramr.infrastructure.training.strategies.diffeomorphic_recon_strategy",
     "DiffeomorphicReconStrategy"),
    ("field_probe_coupled",
     "spectramr.infrastructure.training.strategies.field_probe_coupled_strategy",
     "FieldProbeCoupledStrategy"),
    ("spin_sde",
     "spectramr.infrastructure.training.strategies.spin_sde_strategy",
     "SpinSDEStrategy"),
    ("coord_kspace_gen",
     "spectramr.infrastructure.training.strategies.coord_kspace_gen_strategy",
     "CoordKSpaceGenStrategy"),
    # Integration plan ideas
    ("bloch_bottleneck",
     "spectramr.infrastructure.training.strategies.bloch_bottleneck_strategy",
     "BlochBottleneckStrategy"),
    ("cycle_bloch_digital_twin",
     "spectramr.infrastructure.training.strategies.cycle_bloch_digital_twin_strategy",
     "CycleBlochDigitalTwinStrategy"),
    ("cross_contrast_kspace_diffusion",
     "spectramr.infrastructure.training.strategies.cross_contrast_kspace_diffusion_strategy",
     "CrossContrastKspaceDiffusionStrategy"),
    ("synthetic_pathology_aug",
     "spectramr.infrastructure.training.strategies.synthetic_pathology_aug_strategy",
     "SyntheticPathologyAugStrategy"),
]


@pytest.mark.parametrize("alias,module_path,cls_name", V6_3_STRATEGIES)
def test_module_imports(alias: str, module_path: str, cls_name: str) -> None:
    mod = importlib.import_module(module_path)
    assert hasattr(mod, cls_name)


@pytest.mark.parametrize("alias,module_path,cls_name", V6_3_STRATEGIES)
def test_strategy_factory_resolves(alias: str, module_path: str, cls_name: str) -> None:
    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert alias in paths, f"alias {alias!r} missing"
    assert paths[alias] == f"{module_path}.{cls_name}"


def test_phase2_pipelines_render_to_campaign_dict() -> None:
    """ULF Phase-2 pipelines render as sequential campaign dicts."""
    from spectramr.pipelines.ulf_phase2 import (
        GradientSteeringPipeline,
        KspaceGenSequenceTestPipeline,
        NoiseDominantPipeline,
        OffGridAmorphousPipeline,
        PhaseExtractionThreeTierPipeline,
        SeverelyWarpedPipeline,
    )
    pipelines = [
        NoiseDominantPipeline(),
        OffGridAmorphousPipeline(),
        SeverelyWarpedPipeline(),
        KspaceGenSequenceTestPipeline(),
        PhaseExtractionThreeTierPipeline(),
        GradientSteeringPipeline(),
    ]
    for p in pipelines:
        d = p.to_campaign_dict()
        assert d["execution"] == "sequential"
        # Schema-valid CampaignConfigSchema: stage_groups (not stages).
        assert len(d["stage_groups"]) == 3
        names = [s["name"] for s in d["stage_groups"]]
        assert len(set(names)) == 3
