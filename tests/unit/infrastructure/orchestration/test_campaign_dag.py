"""Tests for the campaign orchestrator's multi-stage DAG support.

Covers:

- Sequential vs. parallel execution mode.
- Topological sort over ``depends_on`` edges.
- Cycle detection.
- Multi-arm parallel-within-stage.
- Phase-2 pipeline → campaign manifest round-trip.
"""

from __future__ import annotations

import pytest

from spectramr.config.schemas.campaign import (
    CampaignConfigSchema,
    CampaignExperimentSchema,
    StageGroupSchema,
)
from spectramr.infrastructure.orchestration.campaign_orchestrator import CampaignOrchestrator


def _experiment(name: str, role: str = "variant", config: str = "x.yaml") -> dict:
    return {"name": name, "config": config, "role": role}


# ---------------------------------------------------------------------------
# Topo sort + cycle detection
# ---------------------------------------------------------------------------


def test_topological_sort_orders_dag() -> None:
    """A → B → C with B.depends_on=[A], C.depends_on=[B] resolves in order."""
    groups = [
        StageGroupSchema.model_validate({
            "name": "C", "depends_on": ["B"],
            "experiments": [_experiment("c")],
        }),
        StageGroupSchema.model_validate({
            "name": "B", "depends_on": ["A"],
            "experiments": [_experiment("b")],
        }),
        StageGroupSchema.model_validate({
            "name": "A",
            "experiments": [_experiment("a")],
        }),
    ]
    orch = CampaignOrchestrator(dry_run=True)
    sorted_groups = orch._topological_sort(groups)
    names = [g.name for g in sorted_groups]
    # Topological order: A before B, B before C.
    assert names.index("A") < names.index("B") < names.index("C")


def test_topological_sort_diamond_resolves() -> None:
    """A → {B, C} → D — diamond DAG, both paths run after A and before D."""
    groups = [
        StageGroupSchema.model_validate({
            "name": "A", "experiments": [_experiment("a")],
        }),
        StageGroupSchema.model_validate({
            "name": "B", "depends_on": ["A"], "experiments": [_experiment("b")],
        }),
        StageGroupSchema.model_validate({
            "name": "C", "depends_on": ["A"], "experiments": [_experiment("c")],
        }),
        StageGroupSchema.model_validate({
            "name": "D", "depends_on": ["B", "C"], "experiments": [_experiment("d")],
        }),
    ]
    orch = CampaignOrchestrator(dry_run=True)
    names = [g.name for g in orch._topological_sort(groups)]
    assert names[0] == "A"
    assert names[-1] == "D"
    assert set(names[1:3]) == {"B", "C"}


def test_topological_sort_detects_cycle() -> None:
    """A → B → A is a cycle; orchestrator must raise."""
    groups = [
        StageGroupSchema.model_validate({
            "name": "A", "depends_on": ["B"], "experiments": [_experiment("a")],
        }),
        StageGroupSchema.model_validate({
            "name": "B", "depends_on": ["A"], "experiments": [_experiment("b")],
        }),
    ]
    orch = CampaignOrchestrator(dry_run=True)
    with pytest.raises(ValueError, match="Cycle detected"):
        orch._topological_sort(groups)


# ---------------------------------------------------------------------------
# Multi-arm parallel within a stage
# ---------------------------------------------------------------------------


def test_parallel_arms_within_stage_all_listed() -> None:
    """A stage with multiple experiments must accept every arm."""
    sg = StageGroupSchema.model_validate({
        "name": "shootout",
        "experiments": [
            _experiment("arm_a", role="variant"),
            _experiment("arm_b", role="variant"),
            _experiment("baseline", role="baseline"),
        ],
    })
    assert len(sg.experiments) == 3
    names = {e.name for e in sg.experiments}
    assert names == {"arm_a", "arm_b", "baseline"}


# ---------------------------------------------------------------------------
# Sequential vs. parallel campaign mode
# ---------------------------------------------------------------------------


def test_sequential_campaign_must_have_stage_groups() -> None:
    with pytest.raises(ValueError, match="stage_groups"):
        CampaignConfigSchema.model_validate({
            "name": "missing_stages",
            "execution": "sequential",
            "experiments": [_experiment("x")],
        })


def test_sequential_campaign_rejects_top_level_experiments() -> None:
    with pytest.raises(ValueError, match="not 'experiments'"):
        CampaignConfigSchema.model_validate({
            "name": "mixed_mode",
            "execution": "sequential",
            "experiments": [_experiment("x")],
            "stage_groups": [{
                "name": "g1",
                "experiments": [_experiment("a")],
            }],
        })


def test_parallel_campaign_accepts_flat_experiments() -> None:
    cfg = CampaignConfigSchema.model_validate({
        "name": "parallel_only",
        "execution": "parallel",
        "experiments": [_experiment("a"), _experiment("b")],
    })
    assert cfg.execution == "parallel"
    assert len(cfg.experiments) == 2


# ---------------------------------------------------------------------------
# Phase-2 pipeline → orchestrator round-trip
# ---------------------------------------------------------------------------


def test_phase2_pipeline_to_campaign_dict_validates() -> None:
    from spectramr.pipelines.ulf_phase2 import (
        GradientSteeringPipeline,
        KspaceGenSequenceTestPipeline,
        NoiseDominantPipeline,
        OffGridAmorphousPipeline,
        PhaseExtractionThreeTierPipeline,
        SeverelyWarpedPipeline,
    )
    for cls in (
        NoiseDominantPipeline,
        OffGridAmorphousPipeline,
        SeverelyWarpedPipeline,
        KspaceGenSequenceTestPipeline,
        PhaseExtractionThreeTierPipeline,
        GradientSteeringPipeline,
    ):
        d = cls().to_campaign_dict()
        # Must validate against the real CampaignConfigSchema.
        cfg = CampaignConfigSchema.model_validate(d)
        # And the topo-sort must succeed (no cycles).
        orch = CampaignOrchestrator(dry_run=True)
        sorted_groups = orch._topological_sort(cfg.stage_groups)
        assert len(sorted_groups) == len(cfg.stage_groups)


def test_phase2_pipeline_with_parallel_arms_renders() -> None:
    """A stage with ``parallel_arms`` produces multiple experiments
    in the same stage group."""
    from spectramr.pipelines.ulf_phase2 import NoiseDominantPipeline

    p = NoiseDominantPipeline()
    # Inject two parallel ablation arms into the second stage.
    p.stages[1].parallel_arms = [
        {
            "name": "ablation_no_score_cond",
            "config": "experiments/inprogress/diffusion/experiment_11_fourier_bridge.yaml",
            "role": "ablation",
        },
        {
            "name": "ablation_no_noise_estimate",
            "config": "experiments/inprogress/diffusion/experiment_11_fourier_bridge.yaml",
            "role": "ablation",
        },
    ]
    cfg = CampaignConfigSchema.model_validate(p.to_campaign_dict())
    middle = cfg.stage_groups[1]
    # Primary + 2 ablation arms = 3 experiments in this stage.
    assert len(middle.experiments) == 3
    roles = {e.role for e in middle.experiments}
    assert "variant" in roles
    assert "ablation" in roles


# ---------------------------------------------------------------------------
# checkpoint_from chain
# ---------------------------------------------------------------------------


def test_checkpoint_from_chain_propagates_through_stages() -> None:
    """A stage with a single dependency emits a checkpoint_from pointing
    back at that dependency's primary experiment."""
    from spectramr.pipelines.ulf_phase2 import OffGridAmorphousPipeline

    p = OffGridAmorphousPipeline()
    cfg = CampaignConfigSchema.model_validate(p.to_campaign_dict())
    # Stage 1+2 each depend on the previous → checkpoint_from must be set.
    for sg in cfg.stage_groups[1:]:
        primary = sg.experiments[0]
        assert primary.checkpoint_from is not None
        assert primary.checkpoint_from.stage in (g.name for g in cfg.stage_groups)
