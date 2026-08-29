"""
Fixtures for critical training variables testing.

This module provides reusable test fixtures for:
- Mock models (generator, discriminator)
- Sample batches (data, labels, masks)
- Physics tensors (k-space, FFT outputs, sensitivity maps)
- Training state (checkpoints, optimizer state, RNG state)
- Device management
- Configuration objects for all 8 training strategies

All fixtures are designed to be portable across strategies and scalable
for parameterized testing across training modes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

# ============================================================================
# DEVICE & HARDWARE FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def device():
    """Get appropriate device for testing (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def amp_dtype():
    """Get appropriate AMP dtype (float16 for CUDA, bfloat16 fallback)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32  # No AMP on CPU


@pytest.fixture
def amp_scaler(device):
    """Create a GradScaler for AMP testing."""
    if device.type == "cuda":
        return GradScaler("cuda")
    return None


# ============================================================================
# MOCK MODELS (Generator, Discriminator, EMA)
# ============================================================================


class SimpleGenerator(nn.Module):
    """Minimal generator for testing (fast, memory-efficient)."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleDiscriminator(nn.Module):
    """Minimal discriminator for testing."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        features = features.flatten(1)
        return self.classifier(features)


@pytest.fixture
def generator(request, device):
    """Create a simple generator on the appropriate device."""
    # Try to get training_strategy from request if it exists (for parameterized tests)
    try:
        training_strategy = request.getfixturevalue("training_strategy")
    except (pytest.FixtureLookupError, ValueError):
        training_strategy = "gan"

    if training_strategy in ("gan", "diffusion", "reconstruction"):
        in_channels = 2
        out_channels = 2
    else:
        in_channels = 1
        out_channels = 1

    model = SimpleGenerator(
        in_channels=in_channels, out_channels=out_channels, hidden_dim=32
    )
    return model.to(device)


@pytest.fixture
def discriminator(device):
    """Create a simple discriminator on the appropriate device."""
    model = SimpleDiscriminator(in_channels=2, hidden_dim=32)
    return model.to(device)


@pytest.fixture
def ema_generator(generator):
    """Create an EMA copy of the generator."""
    import copy

    ema_model = copy.deepcopy(generator)
    return ema_model


# ============================================================================
# DATA BATCH FIXTURES
# ============================================================================


@pytest.fixture
def batch_kspace(device) -> dict[str, torch.Tensor]:
    """Create a sample k-space batch (complex multi-coil)."""
    batch_size = 4
    num_coils = 4
    height, width = 64, 64

    # Complex k-space: (B, C, H, W)
    kspace_real = torch.randn(batch_size, num_coils, height, width, device=device)
    kspace_imag = torch.randn(batch_size, num_coils, height, width, device=device)
    kspace = torch.complex(kspace_real, kspace_imag)

    # RSS image: (B, 1, H, W)
    rss_image = torch.abs(kspace).mean(dim=1, keepdim=True)

    return {
        "kspace": kspace,
        "rss_image": rss_image,
        "batch_size": batch_size,
        "num_coils": num_coils,
    }


@pytest.fixture
def batch_image(device) -> dict[str, torch.Tensor]:
    """Create a sample image batch (magnitude, real-valued)."""
    batch_size = 4
    height, width = 64, 64

    image = torch.randn(batch_size, 1, height, width, device=device).abs()
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return {
        "image": image,
        "batch_size": batch_size,
    }


@pytest.fixture
def batch_with_mask(batch_kspace, device) -> dict[str, torch.Tensor]:
    """Create a batch with undersampling mask."""
    batch_size = 4
    height, width = 64, 64
    center_fraction = 0.08
    acceleration = 4

    # Create Cartesian undersampling mask
    num_center_lines = int(height * center_fraction)
    num_total_lines = height // acceleration
    num_outer_lines = num_total_lines - num_center_lines

    mask = torch.zeros(height, width, device=device)

    # Center region (fully sampled)
    center_start = (height - num_center_lines) // 2
    mask[center_start : center_start + num_center_lines, :] = 1.0

    # Outer region (randomly sampled)
    outer_indices = torch.linspace(
        0, height - 1, height, dtype=torch.long, device=device
    )
    outer_mask = torch.zeros(height, device=device, dtype=torch.bool)
    outer_mask[torch.randperm(height, device=device)[:num_outer_lines]] = True
    mask[outer_mask, :] = 1.0

    # Expand for batch
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    batch = batch_kspace.copy()
    batch["mask"] = mask
    batch["acceleration"] = acceleration

    return batch


@pytest.fixture
def batch_normalized(batch_kspace, device) -> dict[str, torch.Tensor]:
    """Create a normalized batch (useful for metrics testing)."""
    batch = batch_kspace.copy()

    # Normalize k-space by percentile
    kspace_magnitude = torch.abs(batch["kspace"])
    percentile_val = torch.quantile(kspace_magnitude, 0.99)
    batch["kspace_normalized"] = batch["kspace"] / (percentile_val + 1e-8)

    return batch


# ============================================================================
# PHYSICS TENSOR FIXTURES
# ============================================================================


@pytest.fixture
def fft_output(device) -> dict[str, torch.Tensor]:
    """Create FFT output with properties: orthonormality, energy conservation."""
    batch_size = 2
    height, width = 64, 64

    # Create spatial domain image
    image = torch.randn(batch_size, 1, height, width, device=device)

    # Apply centered FFT (ortho normalized)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho", dim=(-2, -1)
        ),
        dim=(-2, -1),
    )

    return {
        "image": image,
        "kspace": kspace,
        "height": height,
        "width": width,
    }


@pytest.fixture
def sensitivity_maps(device) -> torch.Tensor:
    """Create simulated coil sensitivity maps (SENSE)."""
    batch_size = 4
    num_coils = 4
    height, width = 64, 64

    # Generate smooth sensitivity maps using Gaussian kernels
    sens_maps = torch.zeros(
        batch_size, num_coils, height, width, dtype=torch.complex64, device=device
    )

    for b in range(batch_size):
        for c in range(num_coils):
            # Radial sensitivity pattern
            y_grid = torch.linspace(-1, 1, height, device=device)
            x_grid = torch.linspace(-1, 1, width, device=device)
            yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")

            # Gaussian modulated by angle
            phase = np.pi * 2 * c / num_coils
            sensitivity = torch.exp(-(xx**2 + yy**2) / 0.5)
            sensitivity = sensitivity * torch.exp(
                1j * (xx * np.cos(phase) + yy * np.sin(phase))
            )

            sens_maps[b, c] = sensitivity

    return sens_maps


@pytest.fixture
def data_consistency_mask(device) -> torch.Tensor:
    """Create a data consistency projection mask."""
    batch_size = 4
    height, width = 64, 64

    # Binary mask for known k-space locations
    mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=torch.bool)

    # Fill center region (fully sampled)
    center_fraction = 0.08
    center_start = int((height - height * center_fraction) / 2)
    center_end = center_start + int(height * center_fraction)
    mask[:, :, center_start:center_end, :] = True

    return mask


# ============================================================================
# TRAINING STATE & CHECKPOINT FIXTURES
# ============================================================================


@pytest.fixture
def optimizer_state(generator, discriminator, device) -> dict[str, Any]:
    """Create optimizer state for generator and discriminator."""
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Create some gradients by running a fwd-bwd pass
    batch = torch.randn(2, 2, 64, 64, device=device)
    target = torch.randn(2, 2, 64, 64, device=device)

    output_g = generator(batch)
    loss_g = (output_g - target).pow(2).mean()
    loss_g.backward()

    output_d = discriminator(batch)
    loss_d = (output_d - torch.ones_like(output_d)).pow(2).mean()
    loss_d.backward()

    return {
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "lr": 0.0002,
    }


@pytest.fixture
def checkpoint_full(generator, discriminator, optimizer_state) -> dict[str, Any]:
    """Create a complete checkpoint with all required state."""
    return {
        "epoch": 42,
        "iteration": 10000,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_state["optimizer_g"].state_dict(),
        "optimizer_d_state_dict": optimizer_state["optimizer_d"].state_dict(),
        "best_loss": 0.15,
        "learning_rate": optimizer_state["lr"],
        "rng_state_pytorch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "config": {
            "training_mode": "gan",
            "model_type": "standard_unet",
            "batch_size": 32,
        },
    }


@pytest.fixture
def rng_seed():
    """Standard seed for reproducibility tests."""
    return 42


def seed_everything(seed: int, device: torch.device | None = None) -> dict[str, Any]:
    """Seed all RNG sources for deterministic testing.

    Args:
        seed: Random seed value
        device: PyTorch device (if CUDA, seeds CUDA RNG)

    Returns:
        Dictionary with captured RNG states for restoration
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available() and device and device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return {
        "seed": seed,
        "pytorch_rng_state": torch.get_rng_state().cpu().clone(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state().cpu().clone()
            if torch.cuda.is_available()
            else None
        ),
    }


@pytest.fixture
def reproducible_rng(rng_seed, device):
    """Create reproducible RNG state and return seeding function."""
    return seed_everything(rng_seed, device)


# ============================================================================
# CONFIGURATION FIXTURES (FOR ALL 8 TRAINING STRATEGIES)
# ============================================================================


@pytest.fixture
def config_gan() -> dict[str, Any]:
    """Configuration for GAN training (k-space to k-space)."""
    return {
        "training_mode": "gan",
        "data": {
            "source": {"type": "fastmri_h5"},
            "input": {
                "type": "kspace",
                "representation": "complex",
                "coil_processing": "rss",
            },
            "output": {"type": "kspace", "representation": "complex"},
            "batch_size": 32,
            "patch_size": [64, 64],
        },
        "model": {"model_type": "standard_unet", "in_channels": 2, "out_channels": 2},
        "optimization": {"learning_rate": 0.0002},
    }


@pytest.fixture
def config_diffusion() -> dict[str, Any]:
    """Configuration for Diffusion training."""
    return {
        "training_mode": "diffusion",
        "data": {
            "source": {"type": "fastmri_h5"},
            "input": {
                "type": "kspace",
                "representation": "complex",
                "coil_processing": "rss",
            },
            "output": {"type": "kspace", "representation": "complex"},
            "batch_size": 32,
            "physics": {"acceleration": 1},
        },
        "model": {"model_type": "diffusion_unet"},
        "diffusion": {"num_timesteps": 1000},
    }


@pytest.fixture
def config_reconstruction() -> dict[str, Any]:
    """Configuration for Reconstruction training."""
    return {
        "training_mode": "reconstruction",
        "data": {
            "source": {"type": "fastmri_h5"},
            "input": {"type": "kspace", "representation": "complex", "acceleration": 4},
            "output": {"type": "kspace", "representation": "complex"},
            "batch_size": 32,
        },
        "model": {"model_type": "standard_unet", "in_channels": 2, "out_channels": 2},
        "physics": {"data_consistency": True},
    }


@pytest.fixture
def config_vae() -> dict[str, Any]:
    """Configuration for VAE training (image domain)."""
    return {
        "training_mode": "vae",
        "data": {
            "source": {"type": "fastmri_h5"},
            "input": {"type": "image", "representation": "magnitude"},
            "output": {"type": "image", "representation": "magnitude"},
            "batch_size": 32,
        },
        "model": {
            "model_type": "vae",
            "in_channels": 1,
            "out_channels": 1,
            "latent_dim": 128,
        },
        "vae": {"beta": 0.1},
    }


@pytest.fixture
def config_ssl() -> dict[str, Any]:
    """Configuration for SSL training (multi-manifest)."""
    return {
        "training_mode": "ssl",
        "data": {
            "source": {"type": "manifest_roles"},
            "inputs": [
                {
                    "manifest": "fastmri_knee.pkl",
                    "key": "primary",
                    "data_type": "kspace",
                },
                {
                    "manifest": "fastmri_brain.pkl",
                    "key": "auxiliary",
                    "data_type": "image",
                },
            ],
            "batch_size": 32,
        },
        "model": {"model_type": "ssl_encoder", "in_channels": 1, "out_channels": 1},
    }


@pytest.fixture
def config_domain_adaptation() -> dict[str, Any]:
    """Configuration for Domain Adaptation training."""
    return {
        "training_mode": "domain_adaptation",
        "data": {
            "source": {"type": "fastmri_h5"},
            "target": {"type": "nifti"},
            "batch_size": 32,
        },
        "model": {
            "model_type": "da_unet",
            "in_channels": 1,
            "out_channels": 1,
        },
    }


@pytest.fixture
def config_image_translation() -> dict[str, Any]:
    """Configuration for Image Translation (paired NIfTI)."""
    return {
        "training_mode": "gan",
        "data": {
            "source": {"type": "nifti_paired"},
            "input": {"type": "image", "representation": "magnitude"},
            "output": {"type": "image", "representation": "magnitude"},
            "batch_size": 32,
            "patch_size": [320, 320],
        },
        "model": {
            "model_type": "translation_unet",
            "in_channels": 1,
            "out_channels": 1,
        },
    }


@pytest.fixture
def config_latent_diffusion() -> dict[str, Any]:
    """Configuration for Latent Diffusion training."""
    return {
        "training_mode": "latent_diffusion",
        "data": {
            "source": {"type": "fastmri_h5"},
            "input": {"type": "latent_codes", "vae_checkpoint": "checkpoints/vae.pt"},
            "output": {"type": "latent_codes"},
            "batch_size": 32,
        },
        "model": {
            "model_type": "latent_diffusion_unet",
            "in_channels": 1,
            "out_channels": 1,
        },
        "diffusion": {"num_timesteps": 1000},
    }


@pytest.fixture
def all_strategy_configs(
    config_gan,
    config_diffusion,
    config_reconstruction,
    config_vae,
    config_ssl,
    config_domain_adaptation,
    config_image_translation,
    config_latent_diffusion,
) -> dict[str, dict[str, Any]]:
    """Aggregated fixture with all 8 training strategy configs."""
    return {
        "gan": config_gan,
        "diffusion": config_diffusion,
        "reconstruction": config_reconstruction,
        "vae": config_vae,
        "ssl": config_ssl,
        "domain_adaptation": config_domain_adaptation,
        "image_translation": config_image_translation,
        "latent_diffusion": config_latent_diffusion,
    }


# ============================================================================
# TEMPORARY FILE FIXTURES
# ============================================================================


@pytest.fixture
def temp_checkpoint_path(tmp_path) -> Path:
    """Create a temporary path for checkpoint testing."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir / "test_checkpoint.pt"


@pytest.fixture
def temp_config_path(tmp_path) -> Path:
    """Create a temporary path for config testing."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "test_config.yaml"


# ============================================================================
# UTILITY FUNCTIONS FOR TESTS
# ============================================================================


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: tuple | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_device: torch.device | None = None,
    no_nan: bool = True,
    no_inf: bool = True,
):
    """Assert properties of a tensor (shape, dtype, device, NaN/Inf)."""
    if expected_shape is not None:
        assert (
            tensor.shape == expected_shape
        ), f"Shape mismatch: {tensor.shape} vs {expected_shape}"

    if expected_dtype is not None:
        assert (
            tensor.dtype == expected_dtype
        ), f"Dtype mismatch: {tensor.dtype} vs {expected_dtype}"

    if expected_device is not None:
        assert (
            tensor.device == expected_device
        ), f"Device mismatch: {tensor.device} vs {expected_device}"

    if no_nan:
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"

    if no_inf:
        assert not torch.isinf(tensor).any(), "Tensor contains Inf values"


def assert_model_state_dict_identical(
    state_dict_1: dict[str, torch.Tensor],
    state_dict_2: dict[str, torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Assert that two model state dicts are identical."""
    assert set(state_dict_1.keys()) == set(
        state_dict_2.keys()
    ), "State dict keys differ"

    for key in state_dict_1.keys():
        tensor_1 = state_dict_1[key]
        tensor_2 = state_dict_2[key]

        assert torch.allclose(
            tensor_1, tensor_2, rtol=rtol, atol=atol
        ), f"Mismatch in {key}"


def assert_optimizer_state_identical(
    state_dict_1: dict[str, Any],
    state_dict_2: dict[str, Any],
):
    """Assert that two optimizer state dicts are identical."""
    assert (
        state_dict_1.keys() == state_dict_2.keys()
    ), "Optimizer state dict keys differ"

    # Check param groups
    assert len(state_dict_1["param_groups"]) == len(state_dict_2["param_groups"])

    for pg1, pg2 in zip(
        state_dict_1["param_groups"], state_dict_2["param_groups"], strict=False
    ):
        for key in pg1:
            if key == "params":
                continue
            if isinstance(pg1[key], (int, float, str)):
                assert pg1[key] == pg2[key], f"Param group mismatch in {key}"


def compute_batch_statistics(batch: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute statistics for a batch (mean, std, min, max)."""
    if "kspace" in batch:
        tensor = batch["kspace"]
        magnitude = torch.abs(tensor)
    elif "image" in batch:
        tensor = batch["image"]
        magnitude = tensor
    else:
        raise ValueError("Unsupported batch structure")

    return {
        "mean": float(magnitude.mean().item()),
        "std": float(magnitude.std().item()),
        "min": float(magnitude.min().item()),
        "max": float(magnitude.max().item()),
        "max_dynamic_range": float((magnitude.max() / (magnitude.min() + 1e-8)).item()),
    }
