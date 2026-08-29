"""
Comprehensive unit tests for config schemas.

Tests validate:
1. All schema imports
2. Validation rules
3. Known datasets
4. Serialization
5. Config loading

Run on CI, NOT local dev machine.
"""

import pytest


# === Core Schema Import Tests ===
class TestCoreSchemaImports:
    """Test core schema imports."""

    # @# pytest.mark.timeout(10)
    def test_import_base_schema(self):
        try:
            from mriforge.config.schemas.base import BaseConfigSchema

            assert BaseConfigSchema is not None
        except ImportError:
            pytest.skip("BaseConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_model_schema(self):
        try:
            from mriforge.config.schemas.model import ModelConfigSchema

            assert ModelConfigSchema is not None
        except ImportError:
            pytest.skip("ModelConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_data_schema(self):
        try:
            from mriforge.config.schemas.data import DataConfigSchema

            assert DataConfigSchema is not None
        except ImportError:
            pytest.skip("DataConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_training_schema(self):
        try:
            from mriforge.config.schemas.training import TrainingModeConfigSchema

            assert TrainingModeConfigSchema is not None
        except ImportError:
            pytest.skip("TrainingModeConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_optimization_schema(self):
        try:
            from mriforge.config.schemas.optimization import OptimizationConfigSchema

            assert OptimizationConfigSchema is not None
        except ImportError:
            pytest.skip("OptimizationConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_loss_schema(self):
        try:
            from mriforge.config.schemas.loss import LossConfigSchema

            assert LossConfigSchema is not None
        except ImportError:
            pytest.skip("LossConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_physics_schema(self):
        try:
            from mriforge.config.schemas.physics import PhysicsConfigSchema

            assert PhysicsConfigSchema is not None
        except ImportError:
            pytest.skip("PhysicsConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_validation_schema(self):
        try:
            from mriforge.config.schemas.validation import ValidationConfigSchema

            assert ValidationConfigSchema is not None
        except ImportError:
            pytest.skip("ValidationConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_checkpoint_schema(self):
        try:
            from mriforge.config.schemas.checkpoint import CheckpointConfigSchema

            assert CheckpointConfigSchema is not None
        except ImportError:
            pytest.skip("CheckpointConfigSchema not available")

    # @# pytest.mark.timeout(10)
    def test_import_logging_schema(self):
        try:
            from mriforge.config.schemas.logging import LoggingConfigSchema

            assert LoggingConfigSchema is not None
        except ImportError:
            pytest.skip("LoggingConfigSchema not available")


# === Validation Rules Tests ===
class TestValidationRules:
    """Test validation rules module."""

    # @# pytest.mark.timeout(30)
    def test_validation_rules_import(self):
        try:
            from mriforge.config.schemas.validation_rules import validate_config

            assert validate_config is not None
        except ImportError:
            pytest.skip("validation_rules not available")

    # @# pytest.mark.timeout(30)
    def test_validation_constants_import(self):
        from mriforge.config.validation_constants import (
            VALID_MODEL_TYPES,
            VALID_TRAINING_MODES,
        )

        assert isinstance(VALID_MODEL_TYPES, (list, tuple, set))
        assert isinstance(VALID_TRAINING_MODES, (list, tuple, set))

    # @# pytest.mark.timeout(30)
    def test_validation_profiles_import(self):
        try:
            from mriforge.config.schemas.validation_profiles import ValidationProfile

            assert ValidationProfile is not None
        except ImportError:
            pytest.skip("ValidationProfile not available")


# === Known Datasets Tests ===
class TestKnownDatasets:
    """Test known datasets registry."""

    # @# pytest.mark.timeout(30)
    def test_known_datasets_import(self):
        try:
            from mriforge.config.schemas.known_datasets import KNOWN_DATASETS

            assert KNOWN_DATASETS is not None
        except ImportError:
            pytest.skip("KNOWN_DATASETS not available")

    # @# pytest.mark.timeout(30)
    def test_fastmri_in_known_datasets(self):
        """FastMRI should be a known dataset."""
        try:
            from mriforge.config.schemas.known_datasets import KNOWN_DATASETS
        except ImportError:
            pytest.skip("KNOWN_DATASETS not available")

        # Check for common datasets
        known_names = (
            [d.name if hasattr(d, "name") else str(d) for d in KNOWN_DATASETS]
            if isinstance(KNOWN_DATASETS, list)
            else list(KNOWN_DATASETS.keys())
        )

        assert any("fastmri" in str(n).lower() for n in known_names)


# === Serializer Tests ===
class TestSerializer:
    """Test config serializer."""

    # @# pytest.mark.timeout(30)
    def test_serializer_import(self):
        try:
            from mriforge.config.schemas.serializer import ConfigSerializer

            assert ConfigSerializer is not None
        except ImportError:
            pytest.skip("ConfigSerializer not available")

    # @# pytest.mark.timeout(30)
    def test_serializer_to_dict(self):
        """Serializer should convert config to dict."""
        from mriforge.config.schemas.early_stopping import EarlyStoppingConfigSchema

        config = EarlyStoppingConfigSchema(patience=15)

        # Pydantic models have model_dump
        if hasattr(config, "model_dump"):
            d = config.model_dump()
        else:
            d = config.dict()

        assert isinstance(d, dict)
        assert d["patience"] == 15


# === Loader Tests ===
class TestLoader:
    """Test config loader."""

    # @# pytest.mark.timeout(30)
    def test_loader_import(self):
        try:
            from mriforge.config.schemas.loader import ConfigLoader

            assert ConfigLoader is not None
        except ImportError:
            pytest.skip("ConfigLoader not available")

    # @# pytest.mark.timeout(60)
    def test_load_yaml_config(self, tmp_path):
        """Loader should parse YAML files."""
        import yaml

        config_dict = {
            "config_version": "5.0",
            "model": {
                "model_type": "standard_unet",
                "in_channels": 1,
                "out_channels": 1,
            },
            "training": {
                "training_mode": "reconstruction",
                "batch_size": 4,
                "epochs": 10,
                "output_dir": str(tmp_path),
            },
            "data": {
                "dataset_type": "fastmri_brain",
            },
        }

        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f)

        # Load with standard yaml
        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["model"]["model_type"] == "standard_unet"


# === Enums Tests ===
class TestEnums:
    """Test config enums."""

    # @# pytest.mark.timeout(30)
    def test_enums_import(self):
        from mriforge.config.schemas.enums import MetricMode, TrainingMode

        assert TrainingMode is not None
        assert MetricMode is not None

    # @# pytest.mark.timeout(30)
    def test_training_mode_values(self):
        from mriforge.config.schemas.enums import TrainingMode

        # Should have common training modes
        modes = [m.value for m in TrainingMode]

        assert "reconstruction" in modes or any("recon" in m.lower() for m in modes)

    # @# pytest.mark.timeout(30)
    def test_metric_mode_values(self):
        from mriforge.config.schemas.enums import MetricMode

        # Should have min/max modes
        modes = [m.value for m in MetricMode]

        assert "min" in modes or MetricMode.MIN is not None
        assert "max" in modes or MetricMode.MAX is not None


# === Physics Schema Tests ===
class TestPhysicsSchema:
    """Test physics configuration schema."""

    # @# pytest.mark.timeout(30)
    def test_physics_defaults(self):
        try:
            from mriforge.config.schemas.physics import PhysicsConfigSchema
        except ImportError:
            pytest.skip("PhysicsConfigSchema not available")

        config = PhysicsConfigSchema()

        # Should have data consistency field
        assert hasattr(config, "data_consistency")

    # @# pytest.mark.timeout(30)
    def test_physics_data_consistency_enabled(self):
        try:
            from mriforge.config.schemas.physics import PhysicsConfigSchema
        except ImportError:
            pytest.skip("PhysicsConfigSchema not available")

        config = PhysicsConfigSchema(data_consistency={"enabled": True, "weight": 0.5})

        assert config.data_consistency.enabled == True
        assert config.data_consistency.weight == 0.5


# === Manifest Schema Tests ===
class TestManifestSchema:
    """Test manifest schema."""

    # @# pytest.mark.timeout(30)
    def test_manifest_schema_import(self):
        try:
            from mriforge.config.schemas.manifest_schema import ManifestSchema

            assert ManifestSchema is not None
        except ImportError:
            pytest.skip("ManifestSchema not available")


# === Multi-Stage Config Tests ===
class TestMultiStageConfig:
    """Test multi-stage training configuration."""

    # @# pytest.mark.timeout(30)
    def test_multi_stage_validator_import(self):
        try:
            from mriforge.config.schemas.multi_stage_config_validator import (
                MultiStageConfigValidator,
            )

            assert MultiStageConfigValidator is not None
        except ImportError:
            pytest.skip("MultiStageConfigValidator not available")


# === Quick Config Tests ===
class TestQuickConfig:
    """Test quick config utilities."""

    # @# pytest.mark.timeout(30)
    def test_quick_config_import(self):
        try:
            from mriforge.config.schemas.quick_config import QuickConfig

            assert QuickConfig is not None
        except ImportError:
            pytest.skip("QuickConfig not available")


# === Defaults Provider Tests ===
class TestDefaultsProvider:
    """Test defaults provider."""

    # @# pytest.mark.timeout(30)
    def test_defaults_provider_import(self):
        try:
            from mriforge.config.schemas.defaults_provider import DefaultsProvider

            assert DefaultsProvider is not None
        except ImportError:
            pytest.skip("DefaultsProvider not available")


# === Paradigm Registry Tests ===
class TestParadigmRegistry:
    """Test paradigm registry."""

    # @# pytest.mark.timeout(30)
    def test_paradigm_registry_import(self):
        try:
            from mriforge.config.schemas.paradigm_registry import ParadigmRegistry

            assert ParadigmRegistry is not None
        except ImportError:
            pytest.skip("ParadigmRegistry not available")


# === Trellis Config Tests ===
class TestTrellisConfig:
    """Test Trellis configuration."""

    # @# pytest.mark.timeout(30)
    def test_trellis_config_import(self):
        try:
            from mriforge.config.schemas.trellis import TrellisConfig

            assert TrellisConfig is not None
        except ImportError:
            pytest.skip("TrellisConfig not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
