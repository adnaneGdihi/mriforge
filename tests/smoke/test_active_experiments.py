"""
Smoke tests for active experiment configurations.

These tests validate that:
1. Configs load without Pydantic validation errors
2. Model types are registered or can be mocked
3. Strategy classes are importable

NOTE: These tests do NOT run actual training - they validate config structure only.
"""

import glob
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from tests.utils.config_load_baseline import corpus_params


# A ``mock_dependencies`` fixture used to live here. It mocked ``torchkbnuf`` --
# a module that does not exist under that name and that nothing in the tree
# imports; the real package is ``torchkbnufft``, a hard dependency that is
# installed. So the mock protected nothing, and its one real effect was harmful:
# ``patch.dict(sys.modules, ...)`` snapshots ``sys.modules`` on entry and
# restores it on exit, EVICTING every module imported while it was active. At
# module scope that is most of ``spectramr``. The next test file to import those
# modules re-executed them, re-running every ``@register_model`` decorator, and
# the registry correctly refused the duplicate:
#
#     ValueError: Model 'multiscale_latent_discriminator' already registered to
#     ...PatchGANDiscriminator; refusing to overwrite with
#     ...PatchGANDiscriminator      # <- the SAME class, re-imported
#
# That is 160 order-dependent failures across the smoke sweeps (this file first
# => the rest are poisoned; this file second => green), which is why the sharded
# cluster runs that split these files across processes never reported it.


# Helper to import Settings inside tests
def get_settings_class():
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings


# Discover config files in ACTIVE directory
CONFIG_DIR = Path("experiments/active")
CONFIG_FILES = glob.glob(str(CONFIG_DIR / "*.yaml"))
#: Arms recorded as known-unloadable in ``scripts/ci/config_load_baseline.txt``
#: are xfailed rather than failed -- see :mod:`tests.utils.config_load_baseline`.
#: An arm NOT on that list still fails loudly; the ratchet is intact.
CONFIG_PARAMS = corpus_params(CONFIG_FILES)


class DummyModel(torch.nn.Module):
    """Dummy model with trainable parameters for smoke testing."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = "dummy"
        self.dummy_param = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, **kwargs):
        return x


@pytest.fixture(scope="session", autouse=True)
def mock_model_registry():
    """Patch ModelRegistry to return DummyModel for unregistered types."""
    from spectramr.models.factories.model_factory import ModelRegistry

    original_get = ModelRegistry.get_generator_class
    original_has = ModelRegistry.has_generator

    # Models that may not have real implementations or require external deps
    # Note: rectified_flow and consistency_model are now implemented
    DUMMY_MODELS = ["mamba", "kspace_cold_diffusion", "nesvor", "neural_ode"]

    def patched_get(self, name: str):
        cls = original_get(self, name)
        if cls is None or isinstance(cls, MagicMock) or name in DUMMY_MODELS:
            return DummyModel
        return cls

    def patched_has(self, name: str):
        if name in DUMMY_MODELS:
            return True
        return original_has(self, name)

    ModelRegistry.get_generator_class = patched_get
    ModelRegistry.has_generator = patched_has
    yield
    ModelRegistry.get_generator_class = original_get
    ModelRegistry.has_generator = original_has


@pytest.mark.parametrize("config_path", CONFIG_PARAMS)
def test_config_loads_successfully(config_path):
    """Test that config files load without Pydantic validation errors."""
    print(f"\n[SMOKE] Loading config: {config_path}")

    # This validates the entire Pydantic schema
    settings = get_settings_class().from_yaml(config_path)

    # Basic assertions
    assert settings is not None, "Settings should not be None"
    assert settings.model is not None, "Model config should exist"
    assert settings.model.model_type is not None, "model_type should be specified"
    assert settings.training is not None, "Training config should exist"

    print(f"  ✓ Model type: {settings.model.model_type}")
    # v6.0: training section uses strategy_class, not training_mode
    strategy_class = getattr(settings.training, "strategy_class", "unknown")
    task = getattr(settings.training, "task", "unknown")
    print(f"  ✓ Training strategy: {strategy_class or task}")


@pytest.mark.parametrize("config_path", CONFIG_PARAMS)
def test_model_type_registered(config_path, mock_model_registry):
    """Test that model types are registered (or can be mocked)."""
    from spectramr.models.factories.model_factory import ModelFactory

    settings = get_settings_class().from_yaml(config_path)
    model_type = settings.model.model_type

    print(f"\n[SMOKE] Checking model type: {model_type}")

    # Create factory and check registration
    factory = ModelFactory()

    # With our mock, all model types should be "available"
    assert factory._registry.has_generator(model_type) or model_type in [
        "mamba",
        "kspace_cold_diffusion",
        "nesvor",
        "rectified_flow",
        "consistency_model",
        "neural_ode",
        "MambaReconstruction",
        "b0_estimator",
        "disentangled_vae",
    ], f"Model type '{model_type}' is not registered"

    print(f"  ✓ Model type '{model_type}' is available")


@pytest.mark.parametrize("config_path", CONFIG_PARAMS)
def test_strategy_class_importable(config_path):
    """Test that strategy_class can be imported."""
    settings = get_settings_class().from_yaml(config_path)

    strategy_class_path = getattr(settings.training, "strategy_class", None)

    if strategy_class_path is None:
        pytest.skip("No strategy_class specified")

    print(f"\n[SMOKE] Checking strategy: {strategy_class_path}")

    # Parse module and class name
    parts = strategy_class_path.rsplit(".", 1)
    if len(parts) != 2:
        pytest.fail(f"Invalid strategy_class format: {strategy_class_path}")

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
        strategy_cls = getattr(module, class_name)
        assert strategy_cls is not None
        print(f"  ✓ Strategy class '{class_name}' imported successfully")
    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to import strategy class: {e}")


@pytest.mark.parametrize("config_path", CONFIG_PARAMS)
def test_data_config_valid(config_path):
    """Test that data configuration is valid."""
    settings = get_settings_class().from_yaml(config_path)

    if not hasattr(settings, "data") or settings.data is None:
        pytest.skip("No data config")

    print(f"\n[SMOKE] Checking data config for: {config_path}")

    data = settings.data

    # Check dataset_type is valid
    valid_types = [
        "fastmri_kspace",
        "fastmri_knee",
        "fastmri_brain",
        "nifti",
        "nifti_paired",
        "paired_nifti",
        "nifti_paired",
        "contrast_aware_paired",
        "npy_slice",
        "m4raw",
        "3d",
        "kspace",
        "image",
        "dicom",
        "synthetic",
        "graph_mri",
        "nifti_paired",
        "npy_slice",
    ]

    if hasattr(data, "dataset_type"):
        assert data.dataset_type in valid_types, (
            f"Invalid dataset_type: {data.dataset_type}. Must be one of {valid_types}"
        )
        print(f"  ✓ dataset_type: {data.dataset_type}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
