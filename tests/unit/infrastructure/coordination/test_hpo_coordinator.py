"""Unit tests for the HPOCoordinator wiring.

Confirms the coordinator (a) actually delegates to EnhancedHPOptimizer
(no longer a stub), (b) raises early on missing config_path, and (c)
forwards every knob from the request to the optimizer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mriforge.infrastructure.coordination.hpo_coordinator import HPOCoordinator


@pytest.fixture
def coord():
    return HPOCoordinator()


class TestHPOCoordinator:
    def test_raises_on_missing_config_path(self, coord, tmp_path):
        with pytest.raises(ValueError, match="config_path is required"):
            coord.execute_hpo(
                model_types=["unet"],
                input_lr_dir="/tmp",
                input_hr_dir="/tmp",
                output_dir=str(tmp_path),
                config_path=None,
            )

    def test_raises_on_nonexistent_config_path(self, coord, tmp_path):
        with pytest.raises(FileNotFoundError, match="Base config not found"):
            coord.execute_hpo(
                model_types=["unet"],
                input_lr_dir="/tmp",
                input_hr_dir="/tmp",
                output_dir=str(tmp_path),
                config_path="/does/not/exist.yaml",
            )

    @patch("mriforge.models.tuning.enhanced_hyperparameter_tuning.EnhancedHPOptimizer")
    @patch("mriforge.pipelines.hpo.subprocess_training_objective")
    def test_delegates_to_enhanced_optimizer(
        self, mock_obj_factory, mock_optimizer_cls, coord, tmp_path
    ):
        """The coordinator must call EnhancedHPOptimizer.optimize, NOT return
        a placeholder dict (the previous stub behavior)."""
        # Create a real config file so the existence check passes.
        cfg_path = tmp_path / "base.yaml"
        cfg_path.write_text("metadata:\n  name: dummy\n")

        # The optimizer instance returns a structured result.
        mock_optimizer = mock_optimizer_cls.return_value
        mock_optimizer.optimize.return_value = {
            "best_params": {"lr": 1e-4},
            "best_value": 28.5,
            "n_trials": 50,
        }
        mock_obj_factory.return_value = MagicMock()

        results = coord.execute_hpo(
            model_types=["unet", "kspace_cold_diffusion"],
            input_lr_dir="/tmp/data",
            input_hr_dir="/tmp/data",
            output_dir=str(tmp_path / "out"),
            n_trials=10,
            config_path=str(cfg_path),
            objective_metric="val_psnr",
            objective_factory=mock_obj_factory,
        )

        # Both model types ran
        assert set(results.keys()) == {"unet", "kspace_cold_diffusion"}
        for r in results.values():
            assert r["status"] == "completed"
            # Real optimizer payload propagated, NOT the old placeholder
            assert r["best_value"] == 28.5
            assert r["best_params"] == {"lr": 1e-4}

        # EnhancedHPOptimizer instantiated once per model type
        assert mock_optimizer_cls.call_count == 2

    @patch("mriforge.models.tuning.enhanced_hyperparameter_tuning.EnhancedHPOptimizer")
    @patch("mriforge.pipelines.hpo.subprocess_training_objective")
    def test_writes_best_config_yaml_at_completion(
        self, mock_obj_factory, mock_optimizer_cls, coord, tmp_path
    ):
        """After a successful study, best_config.yaml + best_params.json are
        written under <output_dir>/hpo_<model>/ and their paths returned."""
        import json

        import yaml

        # Real base config with structure HPO would mutate
        cfg_path = tmp_path / "base.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(
                {
                    "metadata": {"name": "base"},
                    "optimization": {"learning_rate": 1e-3, "weight_decay": 1e-4},
                    "training": {"output_dir": "/tmp/output"},
                },
                fh,
            )

        mock_optimizer = mock_optimizer_cls.return_value
        mock_optimizer.optimize.return_value = {
            "best_params": {
                "optimization.optimizer.learning_rate": 5e-5,
                "optimization.optimizer.weight_decay": 1e-5,
            },
            "best_value": 28.5,
            "n_trials": 50,
        }
        mock_obj_factory.return_value = MagicMock()

        results = coord.execute_hpo(
            model_types=["unet"],
            input_lr_dir="/tmp",
            input_hr_dir="/tmp",
            output_dir=str(tmp_path / "out"),
            n_trials=10,
            config_path=str(cfg_path),
            objective_metric="val_psnr",
            objective_factory=mock_obj_factory,
        )

        # Returned result includes paths to both artifacts
        unet_result = results["unet"]
        assert "best_config_path" in unet_result
        assert "best_params_path" in unet_result

        best_yaml = Path(unet_result["best_config_path"])
        best_json = Path(unet_result["best_params_path"])
        assert best_yaml.exists()
        assert best_json.exists()

        # YAML carries the winning overrides + audit annotations
        with open(best_yaml) as fh:
            best_cfg = yaml.safe_load(fh)
        assert best_cfg["optimization"]["learning_rate"] == 5e-5
        assert best_cfg["optimization"]["weight_decay"] == 1e-5
        assert best_cfg["metadata"]["hpo_source"] == str(cfg_path)
        assert best_cfg["metadata"]["hpo_objective_metric"] == "val_psnr"
        assert best_cfg["metadata"]["hpo_best_value"] == 28.5
        assert best_cfg["metadata"]["name"].endswith("_hpo_best")

        # output_dir is repointed so follow-up training won't overwrite trials
        assert "_hpo_winner" in best_cfg["training"]["output_dir"]

        # JSON is provenance-grade (raw optuna params)
        with open(best_json) as fh:
            params_record = json.load(fh)
        assert params_record["model_type"] == "unet"
        assert params_record["best_value"] == 28.5
        assert params_record["objective_metric"] == "val_psnr"
        assert params_record["best_params"] == {
            "optimization.optimizer.learning_rate": 5e-5,
            "optimization.optimizer.weight_decay": 1e-5,
        }

    @patch("mriforge.models.tuning.enhanced_hyperparameter_tuning.EnhancedHPOptimizer")
    @patch("mriforge.pipelines.hpo.subprocess_training_objective")
    def test_per_model_failure_is_isolated(
        self, mock_obj_factory, mock_optimizer_cls, coord, tmp_path
    ):
        """If one model's HPO crashes, others still complete."""
        cfg_path = tmp_path / "base.yaml"
        cfg_path.write_text("metadata:\n  name: dummy\n")

        # Optimizer instances raise on the second model only.
        instances = []

        def _factory(*a, **kw):
            inst = MagicMock()
            instances.append(inst)
            if len(instances) == 1:
                inst.optimize.return_value = {"best_params": {}, "best_value": 1.0}
            else:
                inst.optimize.side_effect = RuntimeError("boom")
            return inst

        mock_optimizer_cls.side_effect = _factory
        mock_obj_factory.return_value = MagicMock()

        results = coord.execute_hpo(
            model_types=["unet", "broken_model"],
            input_lr_dir="/tmp",
            input_hr_dir="/tmp",
            output_dir=str(tmp_path / "out"),
            n_trials=5,
            config_path=str(cfg_path),
            objective_factory=mock_obj_factory,
        )

        assert results["unet"]["status"] == "completed"
        assert results["broken_model"]["status"] == "failed"
        assert "boom" in results["broken_model"]["error"]
