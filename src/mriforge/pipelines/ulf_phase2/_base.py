"""Shared base for ULF Phase-2 engineering pipelines.

Each :class:`Phase2Pipeline` renders to a real, schema-valid
``CampaignConfigSchema`` dict — one ``StageGroup`` per pipeline stage,
with a single ``CampaignExperiment`` inside each group that points at a
seed YAML and overrides ``training.training_mode`` to the stage's
strategy alias. The generated manifest is a drop-in for
``python -m mriforge.cli campaign submit ...``.

Each stage is a *parallel within group, sequential between groups*
unit, matching the `docker-compose`-style DAG the campaign runner
already supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStage:
    """One stage of a Phase-2 pipeline.

    Attributes:
        name: short stage identifier (also used as the experiment name).
        strategy: training-mode alias (resolves through STRATEGY_CLASS_PATHS).
        config_overrides: dict of dotted keys → values merged on top of
            ``seed_config`` when this stage runs.
        depends_on: list of upstream stage names that must complete first.
        seed_config: path to the experiment YAML that supplies the
            non-strategy fields (model, data, optimizer, etc.). Defaults
            to a generic recon stub but a pipeline can override per-stage.
        role: campaign role (baseline / variant / ablation).
        slurm_overrides: per-stage SLURM hints (gpus, mem, time, ...).
        parallel_arms: optional list of additional ``CampaignExperimentSchema``
            dicts to run *in parallel* inside this stage group (siblings
            of the primary arm). Use for shootouts within a single
            pipeline stage, e.g. R=4 / R=6 / R=8 sweeps.
    """

    name: str
    strategy: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    seed_config: str = "experiments/inprogress/diffusion/experiment_11_fourier_bridge.yaml"
    role: str = "variant"
    slurm_overrides: dict[str, Any] = field(default_factory=dict)
    parallel_arms: list[dict[str, Any]] = field(default_factory=list)

    def to_experiment_dict(self) -> dict[str, Any]:
        """Render as a ``CampaignExperimentSchema`` dict (single arm)."""
        return {
            "name": self.name,
            "config": self.seed_config,
            "role": self.role,
            "tags": {
                "stage": self.name,
                "training_mode": self.strategy,
            },
            "slurm_overrides": self.slurm_overrides,
        }


@dataclass
class Phase2Pipeline:
    """A named ordered list of stages with sequential dependency semantics."""

    name: str
    description: str
    stages: list[PipelineStage]

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]

    def to_campaign_dict(self) -> dict[str, Any]:
        """Render as a real ``CampaignConfigSchema`` dict.

        - ``execution`` is always ``sequential`` for Phase-2 pipelines.
        - Each stage becomes one ``StageGroup`` with the primary arm and
          any ``parallel_arms`` sharing the same dependency set.
        - Cross-stage checkpoint flow is wired via ``checkpoint_from``
          when a stage has exactly one upstream parent.
        """
        stage_groups: list[dict[str, Any]] = []
        for stage in self.stages:
            primary = stage.to_experiment_dict()
            # checkpoint_from: chain best-checkpoint forward when the
            # stage has exactly one parent. Multiple parents → ambiguous,
            # caller must spell it out.
            if len(stage.depends_on) == 1:
                primary["checkpoint_from"] = {
                    "stage": stage.depends_on[0],
                    "experiment": stage.depends_on[0],
                }
            experiments = [primary] + list(stage.parallel_arms)
            stage_groups.append(
                {
                    "name": stage.name,
                    "description": f"Stage {stage.name} of {self.name}",
                    "depends_on": list(stage.depends_on),
                    "experiments": experiments,
                    "slurm_overrides": stage.slurm_overrides,
                }
            )

        return {
            "name": self.name,
            "description": self.description,
            "execution": "sequential",
            "stage_groups": stage_groups,
        }

    def to_yaml(self, out_path: str) -> str:
        """Write the campaign manifest as YAML and return the path."""
        from pathlib import Path

        import yaml

        d = self.to_campaign_dict()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(yaml.safe_dump(d, sort_keys=False))
        return out_path


__all__ = ["Phase2Pipeline", "PipelineStage"]
