import pytest

from spectramr.config.schemas.multi_stage_config_validator import (
    DependencyList,
    MultiStageConfigValidator,
    MultiStageMode,
    MultiStageTrainingConfig,
    StageType,
)


class TestMultiStageConfigValidator:
    @pytest.fixture
    def validator(self):
        return MultiStageConfigValidator()

    @pytest.fixture
    def valid_config_dict(self):
        return {
            "experiment_name": "test_experiment",
            "mode": "sequential",
            "stages": [
                {
                    "name": "stage1",
                    "model_type": "vae",
                    "epochs": 10,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                },
                {
                    "name": "stage2",
                    "model_type": "diffusion",
                    "dependencies": ["stage1"],
                    "epochs": 20,
                },
            ],
            "data_flow": {"manifest_path": "/path/to/manifest.json"},
        }

    def test_validate_and_normalize_valid(self, validator, valid_config_dict):
        config = validator.validate_and_normalize(valid_config_dict)
        assert isinstance(config, MultiStageTrainingConfig)
        assert config.experiment_name == "test_experiment"
        assert config.mode == MultiStageMode.SEQUENTIAL
        assert len(config.stages) == 2

        stage1 = config.stages[0]
        assert stage1.name == "stage1"
        assert stage1.type == StageType.VAE
        assert stage1.epochs == 10

        stage2 = config.stages[1]
        assert stage2.name == "stage2"
        assert stage2.type == StageType.DIFFUSION
        assert stage2.dependencies == ["stage1"]

    def test_validate_invalid_structure(self, validator):
        with pytest.raises(ValueError, match="Missing required configuration key"):
            validator.validate_and_normalize({})

        with pytest.raises(
            ValueError, match="Configuration must contain at least one stage"
        ):
            validator.validate_and_normalize({"stages": []})

    def test_validate_invalid_mode(self, validator, valid_config_dict):
        valid_config_dict["mode"] = "invalid_mode"
        with pytest.raises(ValueError, match="Invalid mode"):
            validator.validate_and_normalize(valid_config_dict)

    def test_validate_invalid_stage(self, validator, valid_config_dict):
        valid_config_dict["stages"][0].pop("name")
        with pytest.raises(ValueError, match="Invalid stage 0"):
            validator.validate_and_normalize(valid_config_dict)

    def test_validate_unknown_dependency(self, validator, valid_config_dict):
        valid_config_dict["stages"][1]["dependencies"] = ["unknown_stage"]
        with pytest.raises(ValueError, match="depends on unknown stage"):
            validator.validate_and_normalize(valid_config_dict)

    def test_infer_stage_type(self, validator):
        assert validator._infer_stage_type("vqvae") == StageType.VAE
        assert validator._infer_stage_type("ddpm") == StageType.DIFFUSION
        assert validator._infer_stage_type("gan") == StageType.GAN
        assert validator._infer_stage_type("flow") == StageType.FLOW
        assert validator._infer_stage_type("unet") == StageType.RECONSTRUCTION

    def test_save_normalized_config(self, validator, valid_config_dict, tmp_path):
        config = validator.validate_and_normalize(valid_config_dict)
        output_path = tmp_path / "normalized_config.yaml"
        validator.save_normalized_config(config, str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()

        assert "experiment_name: test_experiment" in content
        assert "name: stage1" in content


class TestDependencyList:
    def test_dependency_list(self):
        deps = DependencyList(["stage1"], display=["alias_stage1"])
        assert deps == ["alias_stage1"]
        assert deps.display() == ["alias_stage1"]
        assert str(deps) == "['alias_stage1']"
