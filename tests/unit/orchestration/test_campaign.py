"""Tests for Campaign Schema, State, Ablation Generator, and Evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from spectramr.config.schemas.campaign import (
    AblationAxisSchema,
    CampaignConfigSchema,
    CampaignExperimentSchema,
    EvaluationConfigSchema,
    SlurmDefaultsSchema,
)
from spectramr.infrastructure.orchestration.ablation_config_generator import (
    AblationConfigGenerator,
)
from spectramr.infrastructure.orchestration.campaign_evaluator import (
    CampaignEvaluator,
)
from spectramr.infrastructure.orchestration.campaign_state import (
    CampaignState,
    ExperimentStatus,
)

# ── Schema Tests ─────────────────────────────────────────────────────


class TestCampaignSchema:
    """Test campaign YAML schema parsing."""

    def test_minimal_campaign(self):
        config = CampaignConfigSchema(
            name="test_campaign",
            experiments=[
                CampaignExperimentSchema(name="exp1", config="path/to/exp.yaml"),
            ],
        )
        assert config.name == "test_campaign"
        assert len(config.experiments) == 1
        assert config.output_dir == "experiments/results/campaigns/test_campaign"

    def test_experiment_roles(self):
        config = CampaignConfigSchema(
            name="test",
            experiments=[
                CampaignExperimentSchema(name="base", config="a.yaml", role="baseline"),
                CampaignExperimentSchema(name="v1", config="b.yaml", role="variant"),
                CampaignExperimentSchema(name="v2", config="c.yaml", role="variant"),
                CampaignExperimentSchema(name="abl", config="d.yaml", role="ablation"),
            ],
        )
        assert len(config.baseline_experiments) == 1
        assert len(config.variant_experiments) == 2
        assert len(config.enabled_experiments) == 4

    def test_disabled_experiment(self):
        config = CampaignConfigSchema(
            name="test",
            experiments=[
                CampaignExperimentSchema(name="on", config="a.yaml", enabled=True),
                CampaignExperimentSchema(name="off", config="b.yaml", enabled=False),
            ],
        )
        assert len(config.enabled_experiments) == 1
        assert config.enabled_experiments[0].name == "on"

    def test_config_must_be_yaml(self):
        with pytest.raises(Exception):  # ValidationError
            CampaignExperimentSchema(name="bad", config="path/to/exp.json")

    def test_ablation_axis_label_auto(self):
        axis = AblationAxisSchema(
            config_path="model.model_kwargs.features",
            values=[[32, 64], [64, 128]],
        )
        assert axis.label == "features"

    def test_slurm_defaults(self):
        defaults = SlurmDefaultsSchema()
        # The default partition must be None: the SLURM script generator
        # (slurm_backend.generate_job_script) only emits a ``--partition``
        # directive when one is explicitly requested, because the target cluster
        # has no dedicated "gpu" partition (GPUs come via --gres). A schema
        # default of "gpu" silently overrode that None and made every campaign
        # job emit ``#SBATCH --partition=gpu`` → sbatch rejection. (#15)
        assert defaults.partition is None
        assert defaults.gpus == 1

    def test_evaluation_config_defaults(self):
        ec = EvaluationConfigSchema()
        assert ec.n_bootstrap == 10_000
        assert ec.correction_method == "fdr"

    def test_from_yaml(self, tmp_path: Path):
        data = {
            "name": "yaml_test",
            "experiments": [
                {"name": "exp1", "config": "test.yaml", "role": "baseline"},
            ],
        }
        yaml_path = tmp_path / "campaign.yaml"
        yaml_path.write_text(yaml.dump(data))

        config = CampaignConfigSchema.from_yaml(str(yaml_path))
        assert config.name == "yaml_test"
        assert config.experiments[0].role == "baseline"

    def test_frozen(self):
        config = CampaignConfigSchema(
            name="frozen_test",
            experiments=[CampaignExperimentSchema(name="e", config="e.yaml")],
        )
        with pytest.raises(Exception):  # ValidationError (frozen)
            config.name = "modified"


# ── State Tests ──────────────────────────────────────────────────────


class TestCampaignState:
    """Test campaign state persistence."""

    def test_create_and_save(self, tmp_path: Path):
        state = CampaignState(
            campaign_name="test",
            campaign_dir=str(tmp_path),
            experiments=[
                ExperimentStatus(name="e1", config_path="a.yaml", status="completed"),
                ExperimentStatus(name="e2", config_path="b.yaml", status="running"),
            ],
        )
        saved = state.save()
        assert saved.exists()

        loaded = CampaignState.load(saved)
        assert loaded.campaign_name == "test"
        assert len(loaded.experiments) == 2

    def test_status_properties(self):
        state = CampaignState(
            campaign_name="t",
            campaign_dir="/tmp",
            experiments=[
                ExperimentStatus(name="a", config_path="a.yaml", status="completed", checkpoint_path="/ckpt"),
                ExperimentStatus(name="b", config_path="b.yaml", status="failed"),
                ExperimentStatus(name="c", config_path="c.yaml", status="running"),
                ExperimentStatus(name="d", config_path="d.yaml", status="pending"),
            ],
        )
        assert len(state.completed) == 1
        assert len(state.failed) == 1
        assert len(state.running) == 1
        assert len(state.pending) == 1
        assert not state.all_terminal
        assert state.progress_fraction == 0.5
        assert len(state.evaluable) == 1

    def test_update_experiment(self):
        state = CampaignState(
            campaign_name="t",
            campaign_dir="/tmp",
            experiments=[
                ExperimentStatus(name="a", config_path="a.yaml", status="running"),
            ],
        )
        state.update_experiment("a", status="completed", checkpoint_path="/best.pt")
        assert state.experiments[0].status == "completed"
        assert state.experiments[0].checkpoint_path == "/best.pt"

    def test_summary_table(self):
        state = CampaignState(
            campaign_name="t",
            campaign_dir="/tmp",
            experiments=[
                ExperimentStatus(name="exp", config_path="a.yaml", status="completed"),
            ],
        )
        table = state.summary_table()
        assert "exp" in table
        assert "completed" in table


# ── Ablation Generator Tests ─────────────────────────────────────────


class TestAblationConfigGenerator:
    """Test ablation config generation."""

    def test_grid_mode(self, tmp_path: Path):
        # Create a base config
        base = {
            "metadata": {"name": "base", "description": "test"},
            "training": {"output_dir": "/tmp/out"},
            "model": {"model_kwargs": {"features": [32, 64]}},
            "objectives": {"reconstruction": {"lambda_l1": 1.0}},
        }
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.dump(base))

        axes = [
            AblationAxisSchema(config_path="objectives.reconstruction.lambda_l1", values=[0.5, 1.0, 2.0]),
            AblationAxisSchema(config_path="model.model_kwargs.features", values=[[32, 64], [64, 128]]),
        ]

        out_dir = tmp_path / "ablation"
        configs = AblationConfigGenerator.generate(str(base_path), axes, str(out_dir))

        assert len(configs) == 6  # 3 × 2
        for cfg_path in configs:
            assert Path(cfg_path).exists()
            with open(cfg_path) as f:
                data = yaml.safe_load(f)
            assert "ablation" in data.get("metadata", {}).get("tags", {})

    def test_random_mode(self, tmp_path: Path):
        base = {
            "model": {"lr": 0.001},
            "objectives": {"reconstruction": {"lambda_l1": 1.0}},
        }
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.dump(base))

        axes = [
            AblationAxisSchema(config_path="model.lr", values=[0.0001, 0.001, 0.01, 0.1]),
            AblationAxisSchema(config_path="objectives.reconstruction.lambda_l1", values=[0.5, 1.0, 2.0, 5.0]),
        ]

        configs = AblationConfigGenerator.generate(
            str(base_path), axes, str(tmp_path / "random"),
            mode="random", n_random_samples=3,
        )
        assert len(configs) == 3

    def test_invalid_path_raises(self, tmp_path: Path):
        base = {"model": {"lr": 0.001}}
        base_path = tmp_path / "base.yaml"
        base_path.write_text(yaml.dump(base))

        axes = [AblationAxisSchema(config_path="nonexistent.key", values=[1, 2])]

        with pytest.raises(KeyError):
            AblationConfigGenerator.generate(str(base_path), axes, str(tmp_path / "out"))

    def test_summarise_sweep(self):
        axes = [
            AblationAxisSchema(config_path="a.b", values=[1, 2, 3]),
            AblationAxisSchema(config_path="c.d", values=[0.1, 0.2]),
        ]
        summary = AblationConfigGenerator.summarise_sweep(axes)
        assert "6" in summary  # 3 × 2


# ── Evaluator Tests ──────────────────────────────────────────────────


class TestCampaignEvaluator:
    """Test the cross-experiment evaluation engine."""

    def _make_state_with_metrics(self, tmp_path: Path) -> CampaignState:
        """Create a campaign state with synthetic per-sample metrics."""
        state = CampaignState(
            campaign_name="eval_test",
            campaign_dir=str(tmp_path),
        )
        rng = np.random.default_rng(42)

        for name, role, offset in [
            ("baseline", "baseline", 0.0),
            ("better", "variant", 2.0),
            ("worse", "variant", -1.0),
        ]:
            exp_dir = tmp_path / name
            exp_dir.mkdir()
            npz_path = exp_dir / "per_sample_metrics.npz"

            psnr = rng.normal(30.0 + offset, 2.0, size=50)
            ssim = rng.normal(0.85 + offset * 0.02, 0.05, size=50).clip(0, 1)

            np.savez(npz_path, psnr=psnr, ssim=ssim)

            state.experiments.append(
                ExperimentStatus(
                    name=name,
                    config_path=f"{name}.yaml",
                    role=role,
                    status="completed",
                    checkpoint_path=str(exp_dir / "best.pt"),
                    per_sample_metrics_path=str(npz_path),
                )
            )

        return state

    def test_evaluate_produces_report(self, tmp_path: Path):
        state = self._make_state_with_metrics(tmp_path)
        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(metrics=["psnr", "ssim"], n_bootstrap=100)
        )
        report = evaluator.evaluate(state)

        assert report.campaign_name == "eval_test"
        assert report.leaderboard is not None
        assert not report.leaderboard.empty
        assert len(report.pairwise_results) > 0
        assert report.summary != ""

    def test_leaderboard_ranking(self, tmp_path: Path):
        state = self._make_state_with_metrics(tmp_path)
        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(metrics=["psnr"], n_bootstrap=100)
        )
        report = evaluator.evaluate(state)

        psnr_lb = report.leaderboard[report.leaderboard["metric"] == "psnr"]
        # "better" has +2.0 offset, should rank first
        assert psnr_lb.iloc[0]["experiment"] == "better"

    def test_significance_matrix(self, tmp_path: Path):
        state = self._make_state_with_metrics(tmp_path)
        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(metrics=["psnr"], n_bootstrap=100)
        )
        report = evaluator.evaluate(state)

        assert "psnr" in report.significance_matrix
        sig_df = report.significance_matrix["psnr"]
        assert sig_df.shape == (3, 3)
        # Diagonal should be 1.0
        for i in range(3):
            assert sig_df.iloc[i, i] == pytest.approx(1.0)

    def test_latex_export(self, tmp_path: Path):
        state = self._make_state_with_metrics(tmp_path)
        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(metrics=["psnr", "ssim"], n_bootstrap=100)
        )
        report = evaluator.evaluate(state)

        latex = CampaignEvaluator.leaderboard_to_latex(report.leaderboard)
        assert r"\begin{table}" in latex
        assert "PSNR" in latex
        assert "SSIM" in latex

    def test_report_save_load(self, tmp_path: Path):
        state = self._make_state_with_metrics(tmp_path)
        evaluator = CampaignEvaluator(
            EvaluationConfigSchema(metrics=["psnr"], n_bootstrap=100)
        )
        report = evaluator.evaluate(state)

        json_path = tmp_path / "report.json"
        report.save(json_path)
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["campaign_name"] == "eval_test"
        assert "leaderboard" in data
