"""Shared fixtures and utilities for integration tests.

This file provides common fixtures, helper functions, and test utilities
used across all integration test modules:
- Temporary directories for experiments
- Minimal training/inference configs
- Synthetic data generators
- Model factories
- Checkpoint utilities
- Metrics computation helpers
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from spectramr.config.settings import TrainingSettings

# ============================================================================
# Directory Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def temp_experiment_dir():
    """Create temporary directory for experiment artifacts.

    Cleaned up automatically after test completes.
    Use for: checkpoints, logs, visualizations, configs.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def temp_session_dir():
    """Create temporary directory for entire test session.

    WARNING: Not cleaned up automatically - use for shared artifacts.
    """
    temp_dir = tempfile.mkdtemp(prefix="integration_test_session_")
    yield Path(temp_dir)
    # Cleanup at session end
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def minimal_reconstruction_config_dict(temp_experiment_dir):
    """Minimal reconstruction training config (as dictionary).

    Use for: Quick config creation, modifications before saving.
    """
    return {
        "config_version": "1.0",
        "experiment": {
            "name": "test_reconstruction",
            "output_dir": str(temp_experiment_dir / "checkpoints"),
            "seed": 42,
        },
        "model": {
            "model_type": "standard_unet",
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 16,
            "num_res_units": 1,
        },
        "data": {
            "dataset_type": "synthetic",
            "batch_size": 2,
            "num_workers": 0,
            "image_size": [64, 64],
            "acceleration": 4,
            "center_fraction": 0.08,
            "normalize_kspace": True,
        },
        "optimization": {
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "use_amp": False,
            "optimizer_type": "adam",
            "scheduler_type": "constant",
        },
        "training": {
            "training_mode": "reconstruction",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 1,
            "validate_every_n_epochs": 1,
        },
        "objectives": {
            "reconstruction": {
                "lambda_l1": 1.0,
                "lambda_l2": 0.0,
            },
        },
        "physics": {
            "data_consistency": {
                "enabled": True,
                "train_noise_level": 0.01,
                "eval_noise_level": 0.005,
                "noise_type": "gaussian",
            },
            "kspace": {
                "enforce_hermitian_symmetry": True,
                "enable_kspace_recon": True,
            },
        },
        "checkpoint": {
            "save_top_k": 1,
            "monitor": "val_loss",
            "mode": "min",
            "save_last": True,
        },
        "validation": {
            "compute_image_metrics": True,
            "save_visualizations": False,
        },
    }


@pytest.fixture
def minimal_diffusion_config_dict(temp_experiment_dir):
    """Minimal diffusion training config (as dictionary)."""
    return {
        "config_version": "1.0",
        "experiment": {
            "name": "test_diffusion",
            "output_dir": str(temp_experiment_dir / "checkpoints"),
            "seed": 42,
        },
        "model": {
            "model_type": "kspace_cold_diffusion_generator",
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 16,
            "num_res_units": 1,
        },
        "data": {
            "dataset_type": "synthetic",
            "batch_size": 2,
            "num_workers": 0,
            "image_size": [64, 64],
            "acceleration": 4,
            "normalize_kspace": True,
        },
        "optimization": {
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "use_amp": False,
        },
        "training": {
            "training_mode": "diffusion",
            "diffusion": {
                "num_timesteps": 100,
                "beta_schedule": "linear",
                "cold_diffusion": True,
                "degradation_type": "kspace_undersampling",
            },
        },
        "objectives": {
            "reconstruction": {"lambda_l1": 1.0},
        },
        "physics": {
            "data_consistency": {
                "enabled": True,
                "train_noise_level": 0.01,
                "eval_noise_level": 0.005,
            },
        },
        "checkpoint": {"save_top_k": 1},
    }


@pytest.fixture
def config_factory(temp_experiment_dir):
    """Factory for creating test configs with custom parameters.

    Usage:
        config = config_factory(
            training_mode="reconstruction",
            learning_rate=1e-3,
            batch_size=4,
        )
    """

    def _create_config(**overrides) -> TrainingSettings:
        base_config = {
            "config_version": "1.0",
            "experiment": {
                "name": "test_config",
                "output_dir": str(temp_experiment_dir),
                "seed": 42,
            },
            "model": {
                "model_type": "standard_unet",
                "in_channels": 2,
                "out_channels": 2,
            },
            "data": {"dataset_type": "synthetic", "batch_size": 2},
            "optimization": {"learning_rate": 1e-4, "num_epochs": 1},
            "training": {"training_mode": "reconstruction"},
            "objectives": {"reconstruction": {"lambda_l1": 1.0}},
            "physics": {"data_consistency": {"enabled": True}},
        }

        # Apply overrides
        for key, value in overrides.items():
            # Simple override (nested keys like "optimization.optimizer.learning_rate" not supported)
            if "." in key:
                parts = key.split(".")
                d = base_config
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = value
            else:
                base_config[key] = value

        # Save to temp file
        config_path = temp_experiment_dir / "temp_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(base_config, f)

        return TrainingSettings.from_yaml(str(config_path))

    return _create_config


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def synthetic_kspace_batch():
    """Generate synthetic k-space batch.

    Returns: dict with keys:
        - 'input': Undersampled k-space (B, C=2, H, W)
        - 'target': Fully-sampled k-space (B, C=2, H, W)
        - 'mask': Undersampling mask (B, 1, H, W)
    """
    batch_size = 4
    height, width = 64, 64

    kspace_input = torch.randn(batch_size, 2, height, width)
    kspace_target = torch.randn(batch_size, 2, height, width)

    # Generate 4x acceleration mask (25% sampling)
    mask = torch.zeros(batch_size, 1, height, width)
    mask[:, :, ::4, :] = 1.0  # Sample every 4th line

    return {
        "input": kspace_input,
        "target": kspace_target,
        "mask": mask,
    }


@pytest.fixture
def synthetic_dataloader():
    """Generate synthetic PyTorch DataLoader.

    Returns: DataLoader with 8 samples, batch_size=2.
    Each batch: (kspace_input, kspace_target, mask)
    """
    num_samples = 8
    kspace_data = torch.randn(num_samples, 2, 64, 64)
    targets = torch.randn(num_samples, 2, 64, 64)
    masks = torch.ones(num_samples, 1, 64, 64) * 0.25

    dataset = TensorDataset(kspace_data, targets, masks)
    return DataLoader(dataset, batch_size=2, shuffle=False)


@pytest.fixture
def create_synthetic_dataset():
    """Factory for creating synthetic datasets with custom parameters.

    Usage:
        dataset = create_synthetic_dataset(
            num_samples=16,
            image_size=(128, 128),
            acceleration=8,
        )
    """

    def _create(
        num_samples: int = 8,
        image_size: tuple[int, int] = (64, 64),
        acceleration: int = 4,
        batch_size: int = 2,
    ) -> DataLoader:
        h, w = image_size

        kspace_data = torch.randn(num_samples, 2, h, w)
        targets = torch.randn(num_samples, 2, h, w)

        # Generate mask for acceleration factor
        mask = torch.zeros(num_samples, 1, h, w)
        mask[:, :, ::acceleration, :] = 1.0

        dataset = TensorDataset(kspace_data, targets, mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return _create


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def simple_conv_model():
    """Create simple convolutional model for testing.

    Architecture: Conv2d(2→16) → ReLU → Conv2d(16→2)
    Use for: Quick forward/backward tests without heavy models.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(2, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 2, 3, padding=1),
    )


@pytest.fixture
def create_test_checkpoint(temp_experiment_dir):
    """Factory for creating test checkpoints.

    Usage:
        checkpoint_path = create_test_checkpoint(
            model=my_model,
            epoch=10,
            val_loss=0.123,
        )
    """

    def _create(
        model: torch.nn.Module | None = None,
        epoch: int = 5,
        val_loss: float = 0.1,
        val_psnr: float = 30.0,
        **extra_fields,
    ) -> Path:
        if model is None:
            # Default simple model
            model = torch.nn.Sequential(
                torch.nn.Conv2d(2, 16, 3, padding=1),
                torch.nn.Conv2d(16, 2, 3, padding=1),
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            **extra_fields,
        }

        checkpoint_path = temp_experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    return _create


# ============================================================================
# Metrics Utilities
# ============================================================================


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> float:
    """Compute PSNR between prediction and target.

    Args:
        pred: Predicted tensor (B, C, H, W)
        target: Target tensor (B, C, H, W)
        data_range: Maximum possible value (default: 1.0 for normalized images)

    Returns:
        PSNR in dB
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction

    psnr = 10 * torch.log10(data_range**2 / mse)
    return psnr.item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        MAE (L1 loss)
    """
    mae = torch.nn.functional.l1_loss(pred, target)
    return mae.item()


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error.

    Args:
        pred: Predicted tensor
        target: Target tensor

    Returns:
        MSE (L2 loss)
    """
    mse = torch.nn.functional.mse_loss(pred, target)
    return mse.item()


@pytest.fixture
def metrics_computer():
    """Fixture providing metrics computation functions.

    Usage:
        metrics = metrics_computer(pred, target)
        # Returns: {"psnr": ..., "mae": ..., "mse": ...}
    """

    def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        return {
            "psnr": compute_psnr(pred, target),
            "mae": compute_mae(pred, target),
            "mse": compute_mse(pred, target),
        }

    return _compute_metrics


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
):
    """Assert two tensors are element-wise close.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Custom error message
    """
    assert (
        tensor1.shape == tensor2.shape
    ), f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}. {msg}"

    are_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    if not are_close:
        max_diff = (tensor1 - tensor2).abs().max().item()
        mean_diff = (tensor1 - tensor2).abs().mean().item()
        raise AssertionError(
            f"Tensors not close. Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}. {msg}"
        )


def assert_shape_equals(
    tensor: torch.Tensor, expected_shape: tuple[int, ...], msg: str = ""
):
    """Assert tensor has expected shape.

    Args:
        tensor: Input tensor
        expected_shape: Expected shape tuple
        msg: Custom error message
    """
    assert (
        tensor.shape == expected_shape
    ), f"Shape mismatch: got {tensor.shape}, expected {expected_shape}. {msg}"


def assert_in_range(value: float, min_val: float, max_val: float, msg: str = ""):
    """Assert value is in specified range [min_val, max_val].

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        msg: Custom error message
    """
    assert (
        min_val <= value <= max_val
    ), f"Value {value} not in range [{min_val}, {max_val}]. {msg}"


# ============================================================================
# Config Utilities
# ============================================================================


def save_config_to_yaml(config_dict: dict[str, Any], output_path: Path) -> Path:
    """Save config dictionary to YAML file.

    Args:
        config_dict: Configuration dictionary
        output_path: Path to save YAML file

    Returns:
        Path to saved YAML file
    """
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return output_path


def load_config_from_yaml(config_path: Path) -> TrainingSettings:
    """Load TrainingSettings from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Loaded TrainingSettings instance
    """
    return TrainingSettings.from_yaml(str(config_path))


# ============================================================================
# GPU Utilities
# ============================================================================


@pytest.fixture
def gpu_available():
    """Check if GPU is available.

    Use with @pytest.mark.skipif(not gpu_available) or in test logic.
    """
    return torch.cuda.is_available()


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get torch device (GPU if available and preferred, else CPU).

    Args:
        prefer_gpu: If True, use GPU when available

    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# Seed Utilities
# ============================================================================


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def deterministic_test():
    """Fixture to ensure test runs with deterministic behavior.

    Use when reproducibility is critical (e.g., checking exact outputs).
    """
    set_all_seeds(42)
    yield
    # Reset to non-deterministic after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Timing Utilities
# ============================================================================


class Timer:
    """Simple context manager for timing code blocks.

    Usage:
        with Timer("Model inference"):
            output = model(input)
        # Prints: "Model inference took 0.123 seconds"
    """

    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0.0

    def __enter__(self):
        import time

        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time

        self.end = time.time()
        self.elapsed = self.end - self.start

        if self.verbose:
            print(f"{self.name} took {self.elapsed:.4f} seconds")


@pytest.fixture
def timer():
    """Fixture providing Timer class for performance measurements."""
    return Timer


# ============================================================================
# Cleanup Utilities
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Automatically cleanup CUDA cache after each test (if GPU available).

    Prevents GPU OOM errors in integration tests.
    """
    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Pytest Markers (for reference)
# ============================================================================

# These markers should be defined in pytest.ini, but documented here:
#
# @pytest.mark.slow - Marks test as slow (skipped in fast runs)
# @pytest.mark.gpu - Requires GPU (skipped if CUDA unavailable)
# @pytest.mark.integration - Integration test (runs full pipeline)
# @pytest.mark.smoke - Quick smoke test (< 10 seconds)
