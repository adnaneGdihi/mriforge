"""Artifact-Aware Augmentation Policies for MRI
=============================================

Implements augmentation policies that are aware of common MRI artifacts
and generate more realistic training data for robust super-resolution.
"""

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class MRIArtifactSimulator:
    """Simulator for common MRI artifacts.

    Generates realistic artifacts that occur in clinical MRI scans.
    """

    def __init__(self, img_size: tuple[int, int] = (256, 256)):
        """__init__.

        Args:
            img_size (tuple[int, int]): Description.
        """
        self.img_size = img_size

    def add_rician_noise(
        self,
        image: torch.Tensor,
        sigma: float = 0.05,
    ) -> torch.Tensor:
        """Add Rician noise (characteristic of magnitude MRI images).

        Args:
            image: Input image
            sigma: Noise standard deviation

        Returns:
            Image with Rician noise

        """
        # Generate complex Gaussian noise
        noise_real = torch.randn_like(image) * sigma
        noise_imag = torch.randn_like(image) * sigma

        # Add noise to real and imaginary parts
        real_part = image + noise_real
        imag_part = noise_imag

        # Compute magnitude (Rician distribution)
        noisy_image = torch.sqrt(real_part**2 + imag_part**2)

        return noisy_image

    def add_motion_artifact(
        self,
        image: torch.Tensor,
        severity: float = 0.3,
    ) -> torch.Tensor:
        """Add motion artifacts by simulating patient movement.

        Args:
            image: Input image
            severity: Motion severity (0-1)

        Returns:
            Image with motion artifacts

        """
        batch_size, channels, height, width = image.shape

        # Simulate random motion in k-space using canonical centered FFT
        kspace = fft2c(image)

        # Apply random phase shifts (motion artifact)
        phase_shifts = torch.randn(batch_size, channels, height, width) * severity
        motion_kernel = torch.exp(1j * phase_shifts)

        # Apply motion in k-space
        kspace_motion = kspace * motion_kernel

        # Transform back to image space using canonical centered IFFT
        motion_image = ifft2c(kspace_motion).real

        return motion_image

    def add_gibbs_ringing(
        self,
        image: torch.Tensor,
        cutoff: float = 0.7,
    ) -> torch.Tensor:
        """Add Gibbs ringing artifact from truncated k-space.

        Args:
            image: Input image
            cutoff: k-space cutoff fraction

        Returns:
            Image with Gibbs ringing

        """
        # Transform to k-space using canonical centered FFT
        kspace = fft2c(image)

        # Apply k-space truncation
        h, w = self.img_size
        center_h, center_w = h // 2, w // 2

        # Create truncation mask
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing="ij",
        )
        distances = torch.sqrt(
            ((y_coords - center_h) / (h / 2)) ** 2 + ((x_coords - center_w) / (w / 2)) ** 2,
        )

        mask = (distances <= cutoff).float()
        mask = mask.to(image.device)

        # Apply truncation
        truncated_kspace = kspace * mask.unsqueeze(0).unsqueeze(0)

        # Transform back using canonical centered IFFT
        gibbs_image = ifft2c(truncated_kspace).real

        return gibbs_image

    def add_bias_field(
        self,
        image: torch.Tensor,
        strength: float = 0.2,
    ) -> torch.Tensor:
        """Add intensity bias field artifact.

        Args:
            image: Input image
            strength: Bias field strength

        Returns:
            Image with bias field

        """
        batch_size, channels, height, width = image.shape

        # Create smooth bias field using low-frequency components
        bias_field = torch.zeros_like(image)

        for b in range(batch_size):
            for c in range(channels):
                # Generate low-frequency bias
                bias_low = torch.randn(8, 8).to(image.device)
                bias_low = F.interpolate(
                    bias_low.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

                bias_field[b, c] = bias_low * strength

        # Apply bias field
        biased_image = image * (1 + bias_field)

        return biased_image

    def add_partial_fourier(
        self,
        image: torch.Tensor,
        fraction: float = 0.75,
    ) -> torch.Tensor:
        """Add partial Fourier artifact.

        Args:
            image: Input image
            fraction: Fraction of k-space acquired

        Returns:
            Image with partial Fourier artifact

        """
        # Transform to k-space using canonical centered FFT
        kspace = fft2c(image)

        # Zero out upper half of k-space
        h, w = self.img_size
        kspace[:, :, : int(h * (1 - fraction)), :] = 0

        # Transform back using canonical centered IFFT
        partial_image = ifft2c(kspace).real

        return partial_image


class ArtifactAwareAugmentation:
    """Artifact-aware augmentation policy for MRI data.

    Applies augmentations that respect the physics of MRI artifacts
    and improve model robustness.
    """

    def __init__(
        self,
        artifact_simulator: MRIArtifactSimulator,
        augmentation_config: dict[str, Any] | None = None,
    ):
        """__init__.

        Args:
            artifact_simulator (MRIArtifactSimulator): Description.
            augmentation_config (Optional[dict[str, Any]]): Description.
        """
        self.artifact_simulator = artifact_simulator

        # Default augmentation configuration
        if augmentation_config is None:
            augmentation_config = {
                "rician_noise": {"prob": 0.3, "sigma_range": (0.01, 0.1)},
                "motion_artifact": {"prob": 0.2, "severity_range": (0.1, 0.5)},
                "gibbs_ringing": {"prob": 0.15, "cutoff_range": (0.5, 0.8)},
                "bias_field": {"prob": 0.25, "strength_range": (0.1, 0.3)},
                "partial_fourier": {"prob": 0.1, "fraction_range": (0.6, 0.9)},
                "intensity_scaling": {"prob": 0.4, "scale_range": (0.8, 1.2)},
                "rotation": {"prob": 0.3, "angle_range": (-15, 15)},
                "elastic_deform": {"prob": 0.2, "strength": 0.1},
            }

        self.config = augmentation_config

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply artifact-aware augmentations.

        Args:
            image: Input image

        Returns:
            Augmented image

        """
        augmented = image.clone()

        # Apply artifact-based augmentations
        if random.random() < self.config["rician_noise"]["prob"]:
            sigma = random.uniform(*self.config["rician_noise"]["sigma_range"])
            augmented = self.artifact_simulator.add_rician_noise(augmented, sigma)

        if random.random() < self.config["motion_artifact"]["prob"]:
            severity = random.uniform(*self.config["motion_artifact"]["severity_range"])
            augmented = self.artifact_simulator.add_motion_artifact(augmented, severity)

        if random.random() < self.config["gibbs_ringing"]["prob"]:
            cutoff = random.uniform(*self.config["gibbs_ringing"]["cutoff_range"])
            augmented = self.artifact_simulator.add_gibbs_ringing(augmented, cutoff)

        if random.random() < self.config["bias_field"]["prob"]:
            strength = random.uniform(*self.config["bias_field"]["strength_range"])
            augmented = self.artifact_simulator.add_bias_field(augmented, strength)

        if random.random() < self.config["partial_fourier"]["prob"]:
            fraction = random.uniform(*self.config["partial_fourier"]["fraction_range"])
            augmented = self.artifact_simulator.add_partial_fourier(augmented, fraction)

        # Apply standard augmentations
        if random.random() < self.config["intensity_scaling"]["prob"]:
            scale = random.uniform(*self.config["intensity_scaling"]["scale_range"])
            augmented = augmented * scale

        if random.random() < self.config["rotation"]["prob"]:
            angle = random.uniform(*self.config["rotation"]["angle_range"])
            augmented = self._rotate_image(augmented, angle)

        if random.random() < self.config["elastic_deform"]["prob"]:
            augmented = self._elastic_deform(
                augmented,
                self.config["elastic_deform"]["strength"],
            )

        return augmented

    def _rotate_image(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply rotation augmentation."""
        # Use affine transformation for rotation
        theta = torch.tensor(
            [
                [
                    torch.cos(torch.tensor(angle * np.pi / 180)),
                    -torch.sin(torch.tensor(angle * np.pi / 180)),
                    0,
                ],
                [
                    torch.sin(torch.tensor(angle * np.pi / 180)),
                    torch.cos(torch.tensor(angle * np.pi / 180)),
                    0,
                ],
            ],
            dtype=torch.float,
        ).unsqueeze(0)

        grid = F.affine_grid(theta, image.shape, align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False)

        return rotated

    def _elastic_deform(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Apply elastic deformation."""
        batch_size, channels, height, width = image.shape

        # Create random displacement fields
        dx = torch.randn(batch_size, height, width) * strength
        dy = torch.randn(batch_size, height, width) * strength

        # Smooth displacement fields
        dx = F.conv2d(dx.unsqueeze(1), torch.ones(1, 1, 3, 3) / 9, padding=1).squeeze(1)
        dy = F.conv2d(dy.unsqueeze(1), torch.ones(1, 1, 3, 3) / 9, padding=1).squeeze(1)

        # Create grid
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )

        grid_x = (x_coords + dx).to(image.device)
        grid_y = (y_coords + dy).to(image.device)

        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Apply deformation
        deformed = F.grid_sample(image, grid, align_corners=False)

        return deformed


class RobustAugmentationDataset(torch.utils.data.Dataset):
    """Dataset with artifact-aware augmentations for robust training.

    Applies artifact-aware augmentations to improve model robustness
    to real-world MRI artifacts.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augmentation_policy: ArtifactAwareAugmentation,
        augmentation_prob: float = 0.7,
    ):
        """__init__.

        Args:
            base_dataset (torch.utils.data.Dataset): Description.
            augmentation_policy (ArtifactAwareAugmentation): Description.
            augmentation_prob (float): Description.
        """
        self.base_dataset = base_dataset
        self.augmentation_policy = augmentation_policy
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        """__len__.

        Returns:
            Any: Description.
        """
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get sample from base dataset
        """__getitem__.

        Args:
            idx (Any): Description.
        Returns:
            Any: Description.
        """
        sample = self.base_dataset[idx]

        if isinstance(sample, tuple):
            image, *rest = sample
        else:
            image = sample
            rest = []

        # Apply augmentations with probability
        if random.random() < self.augmentation_prob:
            image = self.augmentation_policy(image)

        # Return augmented sample
        if rest:
            return (image, *rest)
        return image


@register_model(name="artifact_robust_generator", training_mode="reconstruction")
class ArtifactRobustGenerator(nn.Module, IGenerator):
    """Generator with artifact robustness enhancements.

    Incorporates artifact-aware training to improve robustness
    to real-world MRI artifacts.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_generator: IGenerator | None = None,
        artifact_simulator: MRIArtifactSimulator | None = None,
        **kwargs,
    ):
        """Initialize artifact-robust generator.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            base_generator: Optional pre-built generator. If None, builds StandardUNet.
            artifact_simulator: Optional simulator. If None, builds default.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build base generator if not provided
        if base_generator is not None:
            self.base_generator = base_generator
        else:
            from mriforge.models.reconstruction.unet import StandardUNet

            self.base_generator = StandardUNet(
                in_channels=in_channels,
                out_channels=out_channels,
            )

        # Build artifact simulator if not provided
        self.artifact_simulator = artifact_simulator or MRIArtifactSimulator()

        # Artifact detection module
        self.artifact_detector = self._build_artifact_detector()

    def _build_artifact_detector(self) -> nn.Module:
        """Build artifact detection module."""
        return nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 5),  # 5 artifact types
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with artifact awareness.

        forward method for ArtifactRobustGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Generate with base generator
        output = self.base_generator.forward(x)
        return output

    def generate(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """Generate samples."""
        return self.base_generator.generate(batch_size=batch_size, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape."""
        return self.base_generator.get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.base_generator.get_parameter_count() + sum(
            p.numel() for p in self.artifact_detector.parameters()
        )

    @property
    def name(self) -> str:
        """Return model name."""
        return "artifact_robust_generator"


class ArtifactAugmentationFactory:
    """Factory for creating artifact-aware augmentation components."""

    @staticmethod
    def create_artifact_simulator(
        img_size: tuple[int, int] = (256, 256),
    ) -> MRIArtifactSimulator:
        """Create MRI artifact simulator."""
        return MRIArtifactSimulator(img_size)

    @staticmethod
    def create_augmentation_policy(
        artifact_simulator: MRIArtifactSimulator,
        config: dict[str, Any] | None = None,
    ) -> ArtifactAwareAugmentation:
        """Create artifact-aware augmentation policy."""
        return ArtifactAwareAugmentation(artifact_simulator, config)

    @staticmethod
    def create_robust_dataset(
        base_dataset: torch.utils.data.Dataset,
        augmentation_policy: ArtifactAwareAugmentation,
        augmentation_prob: float = 0.7,
    ) -> RobustAugmentationDataset:
        """Create robust augmentation dataset."""
        return RobustAugmentationDataset(
            base_dataset,
            augmentation_policy,
            augmentation_prob,
        )

    @staticmethod
    def create_robust_generator(
        base_generator: IGenerator,
        artifact_simulator: MRIArtifactSimulator,
    ) -> ArtifactRobustGenerator:
        """Create artifact-robust generator."""
        return ArtifactRobustGenerator(base_generator, artifact_simulator)


# Example usage and testing
if __name__ == "__main__":
    # Create artifact simulator
    simulator = ArtifactAugmentationFactory.create_artifact_simulator()

    # Create augmentation policy
    policy = ArtifactAugmentationFactory.create_augmentation_policy(simulator)
    augmentation_policy = policy

    # Test artifact simulation
    test_image = torch.randn(1, 1, 256, 256)

    logger.debug("Testing MRI artifact simulation...")

    # Test individual artifacts
    rician_noisy = simulator.add_rician_noise(test_image)
    motion_artifact = simulator.add_motion_artifact(test_image)
    gibbs_ringing = simulator.add_gibbs_ringing(test_image)
    bias_field = simulator.add_bias_field(test_image)
    partial_fourier = simulator.add_partial_fourier(test_image)

    logger.debug(f"Original shape: {test_image.shape}")
    logger.debug(f"Rician noise shape: {rician_noisy.shape}")
    logger.debug(f"Motion artifact shape: {motion_artifact.shape}")
    logger.debug(f"Gibbs ringing shape: {gibbs_ringing.shape}")
    logger.debug(f"Bias field shape: {bias_field.shape}")
    logger.debug(f"Partial Fourier shape: {partial_fourier.shape}")

    # Test augmentation policy
    augmented = augmentation_policy(test_image)
    logger.debug(f"Augmented shape: {augmented.shape}")

    logger.debug("Artifact-aware augmentation implementation complete!")
