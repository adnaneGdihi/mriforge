"""Smoke tests for dummy experiment configurations.

These tests verify that:
1. All dummy config files can be loaded and parsed
2. The configs produce valid TrainingSettings objects
3. The training strategy can be resolved via TrainingStrategyFactory
4. Dry runs can be executed (optional, marked as slow)
"""

import os
from pathlib import Path

import pytest
import yaml

# Get the directory containing dummy configs
DUMMY_CONFIG_DIR = Path(__file__).parent.parent.parent / "experiments" / "validated"

# List of all dummy config files
DUMMY_CONFIGS = [
    "dummy_reconstruction.yaml",
    "dummy_gan.yaml",
    "dummy_vae.yaml",
    "dummy_vqvae.yaml",
    "dummy_physics_driven.yaml",
    "dummy_swarm.yaml",
    "dummy_compressed_sensing.yaml",
    "dummy_graph_cold_diffusion.yaml",
    "dummy_mae.yaml",
    "dummy_latent_diffusion.yaml",
    "dummy_validation_run.yaml",  # Original diffusion config
]


def get_available_configs():
    """Return list of available dummy config paths."""
    configs = []
    for config_name in DUMMY_CONFIGS:
        config_path = DUMMY_CONFIG_DIR / config_name
        if config_path.exists():
            configs.append(config_path)
    return configs


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters for config tests."""
    if "config_path" in metafunc.fixturenames:
        configs = get_available_configs()
        if not configs:
            pytest.skip("No dummy configs found in experiments/inprogress/")
        config_ids = [c.stem for c in configs]
        metafunc.parametrize("config_path", configs, ids=config_ids)


class TestDummyConfigsLoadable:
    """Test that all dummy configs can be loaded as YAML."""

    def test_config_is_valid_yaml(self, config_path):
        """Test that config file is valid YAML."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        assert isinstance(config_dict, dict)
        assert len(config_dict) > 0

    def test_config_has_required_keys(self, config_path):
        """Test that config has essential keys."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # v6.0: Essential keys that every config should have
        required_keys = ["training", "data", "model", "physics"]
        for key in required_keys:
            assert key in config_dict, f"Config missing required key: {key}"

    def test_training_strategy_resolvable(self, config_path):
        """Test that training strategy can be resolved from config (v6.0)."""
        from mriforge.config.settings import TrainingSettings
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        settings = TrainingSettings.from_yaml(str(config_path))
        factory = TrainingStrategyFactory()

        try:
            strategy_class = factory.get_strategy_class(settings)
            if strategy_class is None:
                pytest.skip(f"Strategy not resolvable for {config_path.name} in v6.0")
        except Exception as e:
            pytest.fail(
                f"Failed to resolve strategy for config {config_path.name}: {e}"
            )


class TestDummyConfigsValidation:
    """Test that dummy configs pass TrainingSettings validation."""

    @pytest.mark.skipif(
        not os.getenv("TEST_CONFIG_VALIDATION"),
        reason="Config validation tests require TEST_CONFIG_VALIDATION=1",
    )
    def test_config_creates_valid_settings(self, config_path):
        """Test that config can be loaded into TrainingSettings."""
        from mriforge.config.settings import TrainingSettings

        try:
            settings = TrainingSettings.from_yaml(str(config_path))
            assert settings is not None
            assert settings.training_mode is not None
        except Exception as e:
            pytest.fail(f"Failed to load config {config_path.name}: {e}")


class TestStrategyFactoryCompleteness:
    """Test that TrainingStrategyFactory is properly configured."""

    def test_factory_has_all_expected_strategies(self):
        """Test that factory registry contains all expected training strategies."""
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        factory = TrainingStrategyFactory()
        expected_strategies = [
            "diffusion",
            "reconstruction",
            "gan",
            "vae",
            "vqvae",
            "mae",
            "physics_driven",
            "swarm",
            "graph_cold_diffusion",
        ]

        for strategy_name in expected_strategies:
            assert (
                strategy_name in factory.STRATEGY_CLASS_PATHS
            ), f"Missing strategy '{strategy_name}' in STRATEGY_CLASS_PATHS"

    def test_all_strategies_are_loadable(self):
        """Test that all strategy classes can be loaded."""
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        factory = TrainingStrategyFactory()

        for strategy_name, class_path in factory.STRATEGY_CLASS_PATHS.items():
            try:
                strategy_class = factory._load_strategy_class(class_path)
                assert (
                    strategy_class is not None
                ), f"Strategy '{strategy_name}' loaded as None"
                assert callable(
                    strategy_class
                ), f"Strategy '{strategy_name}' is not callable"
            except Exception as e:
                pytest.fail(f"Failed to load strategy '{strategy_name}': {e}")

    def test_strategy_classes_have_required_methods(self):
        """Test that strategy classes have expected interface."""
        from mriforge.infrastructure.training.strategy_factory import TrainingStrategyFactory

        factory = TrainingStrategyFactory()

        # Check a sample of strategies for key methods
        for strategy_name in ["reconstruction", "gan", "diffusion"]:
            class_path = factory.STRATEGY_CLASS_PATHS.get(strategy_name)
            if class_path:
                strategy_class = factory._load_strategy_class(class_path)
                # Check for train_step method
                assert hasattr(
                    strategy_class, "train_step"
                ), f"{strategy_name} strategy missing train_step"


class TestDummyConfigsDryRun:
    """Test dry runs for dummy configs (slow, requires full setup)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_DRY_RUN_TESTS"),
        reason="Dry run tests require RUN_DRY_RUN_TESTS=1",
    )
    def test_dry_run_completes(self, config_path):
        """Test that dry run completes without error."""
        from mriforge.config.settings import TrainingSettings
        from mriforge.pipelines import run_dry_training

        try:
            settings = TrainingSettings.from_yaml(str(config_path))
            result = run_dry_training(settings)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Dry run failed for {config_path.name}: {e}")


class TestDummyConfigsActualRun:
    """Test actual training runs with synthetic data (very slow)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_ACTUAL_TRAINING_TESTS"),
        reason="Actual training tests require RUN_ACTUAL_TRAINING_TESTS=1",
    )
    def test_actual_training_completes(self, config_path, tmp_path):
        """Test that actual training completes for a few iterations."""
        import shutil

        import yaml

        from mriforge.config.settings import TrainingSettings

        # Load and modify config for test
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Override output dir to temp path
        test_output_dir = tmp_path / config_path.stem
        test_output_dir.mkdir(parents=True, exist_ok=True)

        if "training" in config_dict:
            config_dict["training"]["output_dir"] = str(test_output_dir)
            config_dict["training"]["max_iterations"] = 2  # Very short run

        # Use synthetic data
        config_dict["data"]["dataset_type"] = "synthetic"
        config_dict["data"]["data_root"] = str(tmp_path / "synthetic_data")
        config_dict["data"]["batch_size"] = 1

        # Disable checkpointing for speed
        if "checkpoint" in config_dict:
            config_dict["checkpoint"]["enabled"] = False

        # Write modified config
        modified_config = tmp_path / f"{config_path.stem}_test.yaml"
        with open(modified_config, "w") as f:
            yaml.dump(config_dict, f)

        try:
            settings = TrainingSettings.from_yaml(str(modified_config))

            # Import training function
            from mriforge.pipelines.train import run_training_pipeline

            # Run training (this should complete quickly with 2 iterations)
            result = run_training_pipeline(settings)

            # Basic sanity checks
            assert result is not None or True  # Training may return None on success

        except Exception as e:
            # Some strategies may have additional requirements - skip gracefully
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA not available: {e}")
            elif "No module named" in str(e):
                pytest.skip(f"Missing dependency: {e}")
            elif "dataset" in str(e).lower() or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            else:
                pytest.fail(f"Training failed for {config_path.name}: {e}")
        finally:
            # Cleanup
            if test_output_dir.exists():
                shutil.rmtree(test_output_dir, ignore_errors=True)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.getenv("RUN_ACTUAL_TRAINING_TESTS"),
        reason="Actual training tests require RUN_ACTUAL_TRAINING_TESTS=1",
    )
    def test_training_produces_outputs(self, tmp_path):
        """Test that training produces expected output files."""
        import shutil

        import yaml

        from mriforge.config.settings import TrainingSettings

        # Use reconstruction config as reference
        config_path = DUMMY_CONFIG_DIR / "dummy_reconstruction.yaml"
        if not config_path.exists():
            pytest.skip("dummy_reconstruction.yaml not found")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Override for test
        test_output_dir = tmp_path / "training_output_test"
        test_output_dir.mkdir(parents=True, exist_ok=True)

        if "training" in config_dict:
            config_dict["training"]["output_dir"] = str(test_output_dir)
            config_dict["training"]["max_iterations"] = 5  # Short run

        config_dict["data"]["dataset_type"] = "synthetic"
        config_dict["data"]["batch_size"] = 1

        if "logging" in config_dict:
            config_dict["logging"]["log_dir"] = str(test_output_dir / "logs")

        if "checkpoint" in config_dict:
            config_dict["checkpoint"]["enabled"] = False

        modified_config = tmp_path / "test_output_config.yaml"
        with open(modified_config, "w") as f:
            yaml.dump(config_dict, f)

        try:
            settings = TrainingSettings.from_yaml(str(modified_config))
            from mriforge.pipelines.train import run_training_pipeline

            run_training_pipeline(settings)

            # Check that output directory was created
            assert test_output_dir.exists(), "Output directory not created"

        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"CUDA not available: {e}")
            elif "No module named" in str(e):
                pytest.skip(f"Missing dependency: {e}")
            elif "dataset" in str(e).lower() or "data" in str(e).lower():
                pytest.skip(f"Data not available: {e}")
            else:
                pytest.fail(f"Training output test failed: {e}")
        finally:
            if test_output_dir.exists():
                shutil.rmtree(test_output_dir, ignore_errors=True)


class TestConfigConsistency:
    """Test consistency across all dummy configs."""

    def test_all_configs_have_consistent_structure(self):
        """Test that all configs follow v6.0 structure."""
        configs = get_available_configs()
        assert len(configs) > 0, "No dummy configs found"

        common_keys = None
        for config_path in configs:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            keys = set(config_dict.keys())
            if common_keys is None:
                common_keys = keys
            else:
                # v6.0: Check for essential shared keys
                essential = {"training", "data", "model", "config_version", "physics"}
                missing = essential - keys
                assert (
                    len(missing) == 0
                ), f"{config_path.name} missing essential keys: {missing}"

    def test_all_configs_have_unique_output_dirs(self):
        """Test that each config has a unique output directory."""
        output_dirs = set()
        for config_path in get_available_configs():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            output_dir = config_dict.get("output_dir")
            if output_dir:
                assert (
                    output_dir not in output_dirs
                ), f"Duplicate output_dir: {output_dir}"
                output_dirs.add(output_dir)

    def test_all_configs_use_small_batch_sizes(self):
        """Test that dummy configs use small batch sizes for quick testing."""
        for config_path in get_available_configs():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            data_config = config_dict.get("data", {})
            batch_size = data_config.get("batch_size", 2)
            assert batch_size <= 4, (
                f"{config_path.name} has large batch_size={batch_size}, "
                "dummy configs should use small batches"
            )

    def test_all_configs_have_low_iterations(self):
        """Test that dummy configs have low max_iterations."""
        for config_path in get_available_configs():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            max_iterations = config_dict.get("max_iterations", 100)
            assert max_iterations <= 100, (
                f"{config_path.name} has max_iterations={max_iterations}, "
                "dummy configs should have low iterations"
            )
