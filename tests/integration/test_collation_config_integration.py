"""Integration tests for collation configuration parsing and selection.

Verifies that:
1. YAML config with collation field is parsed correctly
2. Different configs use different strategies
3. Validation works properly
4. Backward compatibility maintained (configs without collation field)

Author: spectraMR Team
Date: 2026-02-15
"""

import pytest
from pydantic import ValidationError

from spectramr.config.settings import TrainingSettings

BASE_CONFIG = """config_version: '1.0'
data:
  dataset_type: "{dataset_type}"
  data_root: "databases/fastmri"
  batch_size: 2
  {extra_data}
optimization:
  learning_rate: 1e-4
  optimizer_type: "adam"
logging:
  enable_tensorboard: false
training:
  strategy_class: "spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy"
model:
  model_type: "standard_unet"
"""


class TestConfigWithCollation:
    """Test configs that include collation field."""

    def test_minimal_collation_config(self, tmp_path):
        """Config with minimal collation settings should parse"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data='collation:\n    strategy: "image"',
        )
        config_file = tmp_path / "test_minimal.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.collation is not None
        assert config.data.collation.strategy == "image"
        assert config.data.collation.padding_mode == "zeros"
        assert config.data.collation.padding_value == 0.0

    def test_full_collation_config(self, tmp_path):
        """Config with all collation fields should parse correctly"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data="""collation:
    strategy: "image"
    padding_mode: "reflect"
    padding_value: 0.5
    allow_variable_shapes: true
    squeeze_depth_dim: false
    validate_nans: true
    log_strategy_selection: true""",
        )
        config_file = tmp_path / "test_full.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.collation.strategy == "image"
        assert config.data.collation.padding_mode == "reflect"
        assert config.data.collation.padding_value == 0.5
        assert config.data.collation.allow_variable_shapes is True
        assert config.data.collation.squeeze_depth_dim is False

    def test_auto_select_with_null_strategy(self, tmp_path):
        """strategy=null should be valid (triggers auto-selection later)"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data="collation:\n    strategy: null",
        )
        config_file = tmp_path / "test_auto.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.collation.strategy is None

    def test_slab_mode_with_collation(self, tmp_path):
        """enable_slab_mode should work with collation config"""
        config_content = BASE_CONFIG.format(
            dataset_type="nifti",
            extra_data="""enable_slab_mode: true
  patch_size: [256, 256, 1]
  collation:
    strategy: null""",
        )
        config_file = tmp_path / "test_slab.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.sampling.enable_slab_mode is True
        assert config.data.sampling.patch_size == (256, 256, 1)
        assert config.data.collation.strategy is None


class TestConfigWithoutCollation:
    """Test backward compatibility - configs without collation field."""

    def test_old_config_without_collation(self, tmp_path):
        """Config without collation field should use defaults"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data="",  # No collation section
        )
        config_file = tmp_path / "test_old.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.collation is not None
        assert config.data.collation.strategy is None  # Auto-select
        assert config.data.collation.padding_mode == "zeros"


class TestCollationValidation:
    """Test validation of collation fields."""

    def test_invalid_strategy_raises(self, tmp_path):
        """Invalid strategy should raise ValidationError"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data='collation:\n    strategy: "invalid_xyz"',
        )
        config_file = tmp_path / "test_invalid_strategy.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings.from_yaml(str(config_file))

        assert "strategy" in str(exc_info.value).lower()

    def test_invalid_padding_mode_raises(self, tmp_path):
        """Invalid padding_mode should raise ValidationError"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data='collation:\n    padding_mode: "invalid_mode"',
        )
        config_file = tmp_path / "test_invalid_padding.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValidationError) as exc_info:
            TrainingSettings.from_yaml(str(config_file))

        assert "padding_mode" in str(exc_info.value).lower()

    def test_nan_padding_value_raises(self, tmp_path):
        """NaN padding_value should raise ValidationError"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data="collation:\n    padding_value: .nan",
        )
        config_file = tmp_path / "test_nan_padding.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValidationError):
            TrainingSettings.from_yaml(str(config_file))


class TestMultipleConfigs:
    """Test that different configs are independent."""

    @pytest.mark.parametrize(
        "strategy",
        # "universal" removed 2026-08-09: the schema Literal advertised a
        # strategy CollateStrategyFactory cannot build. "physics" is a real
        # one, so this still parametrises over six.
        ["image", "robust", "physics", "slab", "graph", "sequence"],
    )
    def test_strategy_parses_independently(self, tmp_path, strategy):
        """Each strategy should parse independently"""
        config_content = BASE_CONFIG.format(
            dataset_type="kspace",
            extra_data=f'collation:\n    strategy: "{strategy}"',
        )
        config_file = tmp_path / f"test_{strategy}.yaml"
        config_file.write_text(config_content)

        config = TrainingSettings.from_yaml(str(config_file))

        assert config.data.collation.strategy == strategy
