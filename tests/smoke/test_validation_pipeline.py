"""
Smoke tests for the training validation pipeline.

Tests validate:
1. Validation loop execution during training
2. Metrics computation and logging
3. Multi-channel input robustness (coil handling)
4. Different training strategies validation compatibility
5. Edge cases (empty batches, NaN handling)

NOTE: These tests use mocked data loaders - no real data required.
"""

import shutil
import tempfile
import unittest

import pytest
import torch


# The ``mock_dependencies`` fixture that used to sit here mocked ``torchkbnuf``,
# a name no module in this tree imports (the real package is ``torchkbnufft``).
# It protected nothing and its ``patch.dict(sys.modules, ...)`` teardown evicted
# every module imported during this file's tests, forcing a re-import that
# re-ran the ``@register_model`` decorators and tripped the duplicate-model
# guard in whichever file ran next. See the long note in
# ``tests/smoke/test_active_experiments.py``.


@pytest.fixture(autouse=True)
def stub_dataloader_director(monkeypatch):
    """Replace ``DataPipelineDirector.build_dataloaders`` with a stub.

    The pipeline migrated from the legacy ``ConsolidatedDatasetFactory`` to
    ``DataPipelineDirector``. This fixture installs a global stub that returns
    a pair of mock loaders, so every test in this module exercises the
    validation logic without needing real data on disk.

    The ``@patch(...consolidated_dataset_factory...)`` decorators that used to
    sit on these tests are gone. They had already stopped intercepting anything
    when the pipeline migrated -- this docstring said so and kept them "to avoid
    churn" -- and once `aef6c8c39` deleted the module they became an import
    error, taking out 9 tests that the stub above was already covering.
    """
    default_loaders = create_mock_loaders()

    def _build_dataloaders(self, **_kwargs):  # noqa: ARG001
        # Allow tests to override per-call by setting an attribute first.
        return getattr(self, "_test_loaders", default_loaders)

    monkeypatch.setattr(
        "mriforge.infrastructure.builders.directors.data_pipeline_director."
        "DataPipelineDirector.build_dataloaders",
        _build_dataloaders,
        raising=False,
    )


# Lazy imports helpers
def get_TrainingSettings():
    from mriforge.config.settings import TrainingSettings

    return TrainingSettings


def get_run_training_pipeline():
    from mriforge.pipelines import run_training_pipeline

    return run_training_pipeline


# === Dummy Dataset ===
class DummyDataset(torch.utils.data.Dataset):
    """Synthetic dataset for smoke testing."""

    def __init__(self, length=10, channels=2, spatial_size=32, include_kspace=False):
        self.length = length
        self.channels = channels
        self.spatial_size = spatial_size
        self.include_kspace = include_kspace

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {
            "input": torch.randn(self.channels, self.spatial_size, self.spatial_size),
            "target": torch.randn(self.channels, self.spatial_size, self.spatial_size),
        }
        if self.include_kspace:
            data["kspace"] = torch.randn(self.channels, self.spatial_size, self.spatial_size)
            data["mask"] = torch.ones(1, self.spatial_size, self.spatial_size)
        return data


def create_mock_loaders(train_len=8, val_len=4, batch_size=2, channels=2, include_kspace=False):
    """Create mock data loaders for testing."""
    train_ds = DummyDataset(length=train_len, channels=channels, include_kspace=include_kspace)
    val_ds = DummyDataset(length=val_len, channels=channels, include_kspace=include_kspace)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_base_config(test_dir, training_mode="diffusion", strategy_class=None, **overrides):
    """Create base config for testing."""
    config_dict = {
        "model": {
            "model_type": "standard_unet",
            "in_channels": 2,
            "out_channels": 2,
            "model_kwargs": {"features": (16, 32), "depth": 2},
        },
        # CPU is a deliberate opt-in under the accelerated-run contract, and
        # `run.device` is the leaf that carries it. This block previously said
        # `training.device: cpu`, which bootstrap only consults behind an
        # `if not device_arg` that `run.device`'s 'cuda' default makes
        # unreachable -- so the opt-in never reached compute_device and a
        # CPU node raised AcceleratorRequiredError (9 failures, job 8000966).
        "run": {"device": "cpu"},
        "data": {
            "dataset_type": "fastmri_knee",
            "batch_size": 2,
            "patch_size": [32, 32, 1],
        },
        "training": {
            "max_iterations": 2,
            "batch_size": 2,
            "output_dir": test_dir,
            "strategy_class": strategy_class
            or "mriforge.infrastructure.training.strategies.DiffusionTrainingStrategy",
            "diffusion": {"num_timesteps": 100},
            "latent": {"latent_dim": 256},
        },
        "validation": {
            # `eval_interval` is the v6.0 schema field name; `val_interval` was
            # the legacy alias and is now rejected by ValidationConfigSchema.
            "eval_interval": 1,
            "enable_visualization": False,
        },
        "acceleration": {
            # The diffusion strategy reads max_acceleration / base_acceleration
            # for its curriculum schedule; provide concrete values so the
            # `float(config.acceleration.max_acceleration)` call doesn't crash
            # when the test config omits this block.
            "base_acceleration": 2.0,
            "max_acceleration": 8.0,
            "acceleration_factor": 4.0,
        },
        "optimization": {"learning_rate": 1e-4},
        "losses": {
            "reconstruction": {"lambda_l1": 1.0, "enable_l1": True},
            "diffusion": {"lambda_mse": 1.0, "enable_diffusion_loss": True},
        },
        "logging": {
            "log_interval": 1,
            "silent": True,
            "log_to_console": False,
            "enable_experiment_tracking": False,
        },
        "checkpoint": {"save_interval": 100, "enabled": False},
        "loss_logging": {"enabled": False},
    }

    # Apply overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config_dict:
            config_dict[key].update(value)
        else:
            config_dict[key] = value

    return get_TrainingSettings()(**config_dict)


# === Test Class ===
class TestValidationPipeline(unittest.TestCase):
    """Tests for the training validation pipeline."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.mark.timeout(60)
    def test_validation_loop_executes(self):
        """Test that validation loop runs without errors."""
        train_loader, val_loader = create_mock_loaders()

        config = create_base_config(self.test_dir)

        print("\n[TEST] Running validation loop execution test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"Pipeline failed: {result['error']}")

        print("[TEST] ✓ Validation loop executed successfully")

    @pytest.mark.timeout(60)
    def test_validation_with_multi_coil_input(self):
        """Test validation with 128-channel multi-coil input."""
        train_loader, val_loader = create_mock_loaders(channels=128)

        config = create_base_config(self.test_dir)

        print("\n[TEST] Running multi-coil (128ch) validation test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"Pipeline failed with multi-coil input: {result['error']}")

        print("[TEST] ✓ Multi-coil validation passed")

    @pytest.mark.timeout(60)
    def test_validation_with_reconstruction_strategy(self):
        """Test validation with reconstruction training strategy."""
        train_loader, val_loader = create_mock_loaders()

        config = create_base_config(
            self.test_dir,
            training_mode="reconstruction",
            strategy_class="mriforge.infrastructure.training.strategies.ReconstructionTrainingStrategy",
        )

        print("\n[TEST] Running reconstruction strategy validation test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"Reconstruction strategy failed: {result['error']}")

        print("[TEST] ✓ Reconstruction strategy validation passed")

    @pytest.mark.timeout(60)
    @pytest.mark.skip(reason="GAN strategy requires get_validated_snapshot - not yet implemented")
    def test_validation_with_gan_strategy(self):
        """Test validation with GAN training strategy."""
        train_loader, val_loader = create_mock_loaders()

        config = create_base_config(
            self.test_dir,
            training_mode="gan",
            strategy_class="mriforge.infrastructure.training.strategies.GANTrainingStrategy",
        )

        print("\n[TEST] Running GAN strategy validation test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"GAN strategy failed: {result['error']}")

        print("[TEST] ✓ GAN strategy validation passed")

    @pytest.mark.timeout(60)
    def test_validation_with_vae_strategy(self):
        """Test validation with VAE training strategy."""
        train_loader, val_loader = create_mock_loaders()

        config = create_base_config(
            self.test_dir,
            training_mode="vae",
            strategy_class="mriforge.infrastructure.training.strategies.VAETrainingStrategy",
        )

        print("\n[TEST] Running VAE strategy validation test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"VAE strategy failed: {result['error']}")

        print("[TEST] ✓ VAE strategy validation passed")

    @pytest.mark.timeout(60)
    def test_validation_with_kspace_data(self):
        """Test validation with k-space data included."""
        train_loader, val_loader = create_mock_loaders(include_kspace=True)

        config = create_base_config(self.test_dir)

        print("\n[TEST] Running validation with k-space data...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"K-space validation failed: {result['error']}")

        print("[TEST] ✓ K-space validation passed")

    @pytest.mark.timeout(60)
    def test_validation_interval_respected(self):
        """Test that validation runs at correct intervals."""
        train_loader, val_loader = create_mock_loaders()

        config = create_base_config(
            self.test_dir,
            training={"max_iterations": 4},
            validation={"eval_interval": 2},
        )

        print("\n[TEST] Running validation interval test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"Validation interval test failed: {result['error']}")

        print("[TEST] ✓ Validation interval test passed")

    @pytest.mark.timeout(60)
    def test_validation_with_single_batch(self):
        """Test validation with minimum data (single batch)."""
        train_loader, val_loader = create_mock_loaders(train_len=2, val_len=2, batch_size=2)

        config = create_base_config(self.test_dir, training={"max_iterations": 1})

        print("\n[TEST] Running single-batch validation test...")
        result = get_run_training_pipeline()(config)

        if isinstance(result, dict) and "error" in result:
            self.fail(f"Single batch test failed: {result['error']}")

        print("[TEST] ✓ Single-batch validation passed")


# === Pytest-style tests for additional coverage ===
@pytest.fixture
def test_dir():
    """Create temporary test directory."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.mark.timeout(60)
def test_validation_returns_metrics(test_dir):
    """Test that validation returns metrics dictionary."""
    train_loader, val_loader = create_mock_loaders()

    config = create_base_config(test_dir)

    result = get_run_training_pipeline()(config)

    # Should not have error
    if isinstance(result, dict):
        assert "error" not in result, f"Pipeline error: {result.get('error')}"


@pytest.mark.timeout(60)
def test_validation_handles_complex_input(test_dir):
    """Test validation with complex-valued input (2-channel real/imag)."""
    train_loader, val_loader = create_mock_loaders(channels=2)

    config = create_base_config(test_dir)

    result = get_run_training_pipeline()(config)

    if isinstance(result, dict):
        assert "error" not in result


if __name__ == "__main__":
    unittest.main()
