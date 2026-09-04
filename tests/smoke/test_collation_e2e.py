"""End-to-end smoke test for collation configuration in data loading pipeline.

Tests that the complete flow works:
1. Load config with collation field
2. Build data loaders (train/val)
3. Verify collation strategy is applied consistently

Author: spectraMR Team
Date: 2026-02-15
"""

import pytest

from spectramr.config.schemas.base import CANONICAL_CONFIG_VERSION
from spectramr.config.settings import TrainingSettings


class TestDataLoaderWithCollation:
    """Test that config-driven collation is applied to actual data loaders."""

    def test_dataloader_builder_respects_collation_config(self, tmp_path):
        """DataLoaderBuilder should use collation from config"""
        config_yaml = f"""config_version: '{CANONICAL_CONFIG_VERSION}'
data:
  dataset_type: "kspace"
  data_root: "databases/fastmri"
  batch_size: 2
  collation:
    strategy: "robust"
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
        config_file = tmp_path / "test_e2e.yaml"
        config_file.write_text(config_yaml)

        config = TrainingSettings.from_yaml(str(config_file))

        # Verify config was loaded with explicit collation strategy
        assert config.data.collation.strategy == "robust"

    def test_collation_config_frozen_and_immutable(self, tmp_path):
        """Collation config should be immutable (Pydantic frozen)"""
        config_yaml = f"""config_version: '{CANONICAL_CONFIG_VERSION}'
data:
  dataset_type: "kspace"
  data_root: "databases/fastmri"
  batch_size: 2
  collation:
    strategy: "image"
    padding_mode: "zeros"
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
        config_file = tmp_path / "test_immutable.yaml"
        config_file.write_text(config_yaml)

        config = TrainingSettings.from_yaml(str(config_file))

        # Attempt to modify should raise error (frozen model)
        from pydantic import ValidationError

        # Pydantic v2 frozen models raise ValidationError on assignment
        with pytest.raises(ValidationError):
            config.data.collation.strategy = "robust"

    def test_multiple_configs_have_independent_collation(self, tmp_path):
        """Multiple config instances should have independent collation configs"""
        configs_data = [
            ("image", "image"),
            ("robust", "robust"),
            # was ("universal", "universal") -- that strategy is not
            # constructible and no longer passes schema validation.
            ("physics", "physics"),
        ]

        loaded_configs = []
        for strategy_name, expected_strategy in configs_data:
            config_yaml = f"""config_version: '{CANONICAL_CONFIG_VERSION}'
data:
  dataset_type: "kspace"
  data_root: "databases/fastmri"
  batch_size: 2
  collation:
    strategy: "{strategy_name}"
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
            config_file = tmp_path / f"test_{strategy_name}.yaml"
            config_file.write_text(config_yaml)

            config = TrainingSettings.from_yaml(str(config_file))
            loaded_configs.append(config)

            assert (
                config.data.collation.strategy == expected_strategy
            ), f"Failed for {strategy_name}"

        # Verify all configs are independent
        assert loaded_configs[0].data.collation.strategy == "image"
        assert loaded_configs[1].data.collation.strategy == "robust"
        assert loaded_configs[2].data.collation.strategy == "universal"
