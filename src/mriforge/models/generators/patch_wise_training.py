"""Patch-wise Memory-Efficient Training and On-the-Fly Inference

This module implements memory-efficient training and inference strategies
using overlapping patches with boundary blending for high-resolution MRI
super-resolution on limited GPU memory.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class PatchConfig:
    """Configuration for patch-based processing."""

    patch_size: int = 128
    overlap: int = 32
    batch_size: int = 4
    min_patch_size: int = 64
    max_patch_size: int = 256
    size_variation: bool = True
    blend_mode: str = "gaussian"  # 'gaussian', 'linear', 'cosine'


@dataclass
class InferenceConfig:
    """Configuration for inference processing."""

    tile_size: int = 512
    overlap: int = 64
    batch_size: int = 1
    blend_strength: float = 0.5
    use_tta: bool = False  # Test-time augmentation


class PatchExtractor:
    """Extract overlapping patches from images for memory-efficient processing.

    Supports variable patch sizes and overlap-tile strategy for inference.
    """

    def __init__(self, config: PatchConfig):
        """__init__.

        Args:
            config (PatchConfig): Description.
        """
        self.config = config
        self.patch_cache = {}

    def extract_patches(
        self,
        image: torch.Tensor,
        patch_size: int | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Extract overlapping patches from input image.

        Args:
            image: Input image tensor [C, H, W]
            patch_size: Override default patch size

        Returns:
            Tuple of (patches_tensor, patch_info_list)

        """
        if patch_size is None:
            patch_size = self.config.patch_size

        C, H, W = image.shape
        overlap = self.config.overlap
        stride = patch_size - overlap

        patches = []
        patch_info = []

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Handle edge cases
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)

                patch = image[:, y:y_end, x:x_end]

                # Pad if necessary
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    pad_h = patch_size - patch.shape[1]
                    pad_w = patch_size - patch.shape[2]
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, pad_w, 0, pad_h),
                        mode="reflect",
                    )

                patches.append(patch)

                patch_info.append(
                    {
                        "y_start": y,
                        "x_start": x,
                        "y_end": y_end,
                        "x_end": x_end,
                        "padded": (patch.shape[1] != y_end - y or patch.shape[2] != x_end - x),
                        "original_shape": (y_end - y, x_end - x),
                    },
                )

        if not patches:
            # Handle case where image is smaller than patch
            patch = torch.nn.functional.pad(
                image,
                (0, patch_size - W, 0, patch_size - H),
                mode="reflect",
            )
            patches = [patch]
            patch_info = [
                {
                    "y_start": 0,
                    "x_start": 0,
                    "y_end": H,
                    "x_end": W,
                    "padded": True,
                    "original_shape": (H, W),
                },
            ]

        return torch.stack(patches), patch_info

    def reconstruct_image(
        self,
        patches: torch.Tensor,
        patch_info: list[dict[str, Any]],
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct image from patches with boundary blending.

        Args:
            patches: Processed patches [N, C, H, W]
            patch_info: Patch location information
            target_shape: Target image shape (H, W)

        Returns:
            Reconstructed image [C, H, W]

        """
        C = patches.shape[1]
        reconstructed = torch.zeros(
            (C, target_shape[0], target_shape[1]),
            dtype=patches.dtype,
            device=patches.device,
        )
        weight_mask = torch.zeros_like(reconstructed)

        for i, info in enumerate(patch_info):
            y_start, x_start = info["y_start"], info["x_start"]
            y_end, x_end = info["y_end"], info["x_end"]
            original_h, original_w = info["original_shape"]

            patch = patches[i]

            # Remove padding if applied
            if info["padded"]:
                patch = patch[:, :original_h, :original_w]

            # Create blending weights
            blend_weights = self._create_blend_weights(
                (original_h, original_w),
                self.config.blend_mode,
            )

            # Accumulate patch with weights
            reconstructed[:, y_start:y_end, x_start:x_end] += patch * blend_weights
            weight_mask[:, y_start:y_end, x_start:x_end] += blend_weights

        # Normalize by weights
        reconstructed = torch.where(
            weight_mask > 0,
            reconstructed / weight_mask,
            reconstructed,
        )

        return reconstructed

    def _create_blend_weights(
        self,
        patch_shape: tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        """Create blending weights for patch boundaries.

        Args:
            patch_shape: Shape of the patch (H, W)
            mode: Blending mode ('gaussian', 'linear', 'cosine')

        Returns:
            Weight tensor [H, W]

        """
        H, W = patch_shape
        y_coords = torch.linspace(-1, 1, H)
        x_coords = torch.linspace(-1, 1, W)

        Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

        if mode == "gaussian":
            # Gaussian falloff from center
            sigma = 0.5
            weights = torch.exp(-0.5 * (Y**2 + X**2) / sigma**2)
        elif mode == "cosine":
            # Cosine blending
            weights = 0.5 * (torch.cos(torch.pi * Y) + 1) * 0.5 * (torch.cos(torch.pi * X) + 1)
        else:  # linear
            # Linear blending
            weights = (1 - torch.abs(Y)) * (1 - torch.abs(X))

        return weights


class VariablePatchSampler:
    """Sampler for variable-sized patches during training.

    Enables mixed-size training patches for better generalization.
    """

    def __init__(self, config: PatchConfig):
        """__init__.

        Args:
            config (PatchConfig): Description.
        """
        self.config = config
        self.size_distribution = self._create_size_distribution()

    def _create_size_distribution(self) -> dict[int, float]:
        """Create probability distribution for different patch sizes."""
        sizes = list(
            range(
                self.config.min_patch_size,
                self.config.max_patch_size + 1,
                32,  # Step size
            ),
        )

        if self.config.size_variation:
            # Favor medium sizes with some variation
            center = (self.config.min_patch_size + self.config.max_patch_size) // 2
            probs = []
            for size in sizes:
                # Gaussian distribution around center
                prob = np.exp(-0.5 * ((size - center) / (center * 0.3)) ** 2)
                probs.append(prob)

            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]
        else:
            # Uniform distribution
            probs = [1.0 / len(sizes)] * len(sizes)

        return dict(zip(sizes, probs, strict=False))

    def sample_patch_size(self) -> int:
        """Sample a patch size from the distribution."""
        sizes = list(self.size_distribution.keys())
        probs = list(self.size_distribution.values())
        return np.random.choice(sizes, p=probs)

    def get_batch_patch_sizes(self, batch_size: int) -> list[int]:
        """Sample patch sizes for a batch."""
        return [self.sample_patch_size() for _ in range(batch_size)]


class MemoryEfficientTrainer:
    """Memory-efficient trainer using patch-based processing.

    Processes large images in patches to fit within GPU memory constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_config: PatchConfig,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    ):
        """__init__.

        Args:
            model (nn.Module): Description.
            patch_config (PatchConfig): Description.
            device (torch.device): Description.
        """
        self.model = model.to(device)
        self.device = device
        self.patch_extractor = PatchExtractor(patch_config)
        self.patch_sampler = VariablePatchSampler(patch_config)
        self.config = patch_config

    def train_step(
        self,
        hr_batch: torch.Tensor,
        lr_batch: torch.Tensor,
    ) -> dict[str, float]:
        """Perform training step with patch-based processing.

        Args:
            hr_batch: High-resolution batch [B, C, H, W]
            lr_batch: Low-resolution batch [B, C, H, W]

        Returns:
            Dictionary of training metrics

        """
        batch_size = hr_batch.shape[0]
        total_loss = 0.0
        processed_patches = 0

        # Sample different patch sizes for this batch
        patch_sizes = self.patch_sampler.get_batch_patch_sizes(batch_size)

        for i in range(batch_size):
            hr_image = hr_batch[i : i + 1]  # [1, C, H, W]
            lr_image = lr_batch[i : i + 1]  # [1, C, H, W]
            patch_size = patch_sizes[i]

            # Extract patches
            hr_patches, hr_info = self.patch_extractor.extract_patches(
                hr_image.squeeze(0),
                patch_size,
            )
            lr_patches, lr_info = self.patch_extractor.extract_patches(
                lr_image.squeeze(0),
                patch_size,
            )

            # Process patches in smaller batches
            patch_batch_size = min(self.config.batch_size, len(hr_patches))

            for j in range(0, len(hr_patches), patch_batch_size):
                hr_patch_batch = hr_patches[j : j + patch_batch_size].to(self.device)
                lr_patch_batch = lr_patches[j : j + patch_batch_size].to(self.device)

                # Forward pass
                with torch.amp.autocast():
                    output = self.model(lr_patch_batch)
                    loss = nn.functional.mse_loss(output, hr_patch_batch)

                # Backward pass (assuming optimizer is handled externally)
                loss.backward()

                total_loss += loss.item() * len(hr_patch_batch)
                processed_patches += len(hr_patch_batch)

        return {
            "loss": total_loss / processed_patches,
            "patches_processed": processed_patches,
            "avg_patch_size": sum(patch_sizes) / len(patch_sizes),
        }


class OnTheFlyInference:
    """On-the-fly inference with overlap-tile strategy.

    Enables processing of large images that don't fit in GPU memory.
    """

    def __init__(self, config: InferenceConfig):
        """__init__.

        Args:
            config (InferenceConfig): Description.
        """
        self.config = config
        self.patch_extractor = PatchExtractor(
            PatchConfig(
                patch_size=config.tile_size,
                overlap=config.overlap,
                batch_size=config.batch_size,
            ),
        )

    def infer_large_image(
        self,
        model: nn.Module,
        lr_image: torch.Tensor,
        hr_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Perform inference on large image using overlap-tile strategy.

        Args:
            model: Super-resolution model
            lr_image: Low-resolution input [C, H, W]
            hr_shape: Target high-resolution shape (optional)

        Returns:
            High-resolution output [C, H_hr, W_hr]

        """
        device = next(model.parameters()).device

        # Extract patches
        lr_patches, patch_info = self.patch_extractor.extract_patches(lr_image)

        # Process patches
        hr_patches = []
        for i in range(0, len(lr_patches), self.config.batch_size):
            batch = lr_patches[i : i + self.config.batch_size].to(device)

            with torch.no_grad():
                if self.config.use_tta:
                    # Test-time augmentation
                    hr_batch = self._tta_inference(model, batch)
                else:
                    hr_batch = model(batch)

            hr_patches.extend(hr_batch.cpu())

        hr_patches = torch.stack(hr_patches)

        # Determine target shape
        if hr_shape is None:
            # Assume 4x upsampling
            _, H_lr, W_lr = lr_image.shape
            hr_shape = (H_lr * 4, W_lr * 4)

        # Reconstruct image with blending
        hr_image = self.patch_extractor.reconstruct_image(
            hr_patches,
            patch_info,
            hr_shape,
        )

        return hr_image

    def _tta_inference(self, model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """Test-time augmentation inference.

        Args:
            model: Super-resolution model
            batch: Input batch [B, C, H, W]

        Returns:
            Augmented predictions [B, C, H*scale, W*scale]

        """
        # Original
        pred1 = model(batch)

        # Horizontal flip
        pred2 = model(torch.flip(batch, dims=[3]))
        pred2 = torch.flip(pred2, dims=[3])

        # Vertical flip
        pred3 = model(torch.flip(batch, dims=[2]))
        pred3 = torch.flip(pred3, dims=[2])

        # Both flips
        pred4 = model(torch.flip(batch, dims=[2, 3]))
        pred4 = torch.flip(pred4, dims=[2, 3])

        # Average predictions
        return (pred1 + pred2 + pred3 + pred4) / 4


def create_memory_efficient_trainer(
    model: nn.Module,
    patch_size: int = 128,
    overlap: int = 32,
) -> MemoryEfficientTrainer:
    """Factory function for memory-efficient trainer.

    Args:
        model: Neural network model
        patch_size: Size of training patches
        overlap: Overlap between patches

    Returns:
        Configured MemoryEfficientTrainer

    """
    config = PatchConfig(
        patch_size=patch_size,
        overlap=overlap,
        batch_size=4,
        size_variation=True,
    )

    return MemoryEfficientTrainer(model, config)


def create_otf_inference(
    tile_size: int = 512,
    overlap: int = 64,
    use_tta: bool = False,
) -> OnTheFlyInference:
    """Factory function for on-the-fly inference.

    Args:
        tile_size: Size of inference tiles
        overlap: Overlap between tiles
        use_tta: Whether to use test-time augmentation

    Returns:
        Configured OnTheFlyInference

    """
    config = InferenceConfig(
        tile_size=tile_size,
        overlap=overlap,
        batch_size=1,
        use_tta=use_tta,
    )

    return OnTheFlyInference(config)


# Example usage and testing
if __name__ == "__main__":
    # Test patch extraction and reconstruction
    logger.debug("Testing Patch-wise Processing:")

    # Create test image
    test_image = torch.rand(1, 256, 256)  # [C, H, W]

    # Test patch extractor
    config = PatchConfig(patch_size=128, overlap=32)
    extractor = PatchExtractor(config)

    patches, patch_info = extractor.extract_patches(test_image.squeeze(0))
    logger.debug(f"Original image shape: {test_image.shape}")
    logger.debug(f"Number of patches: {len(patches)}")
    logger.debug(f"Patch shape: {patches[0].shape}")

    # Test reconstruction
    reconstructed = extractor.reconstruct_image(patches, patch_info, (256, 256))
    logger.debug(f"Reconstructed shape: {reconstructed.shape}")

    # Test variable patch sampler
    sampler = VariablePatchSampler(config)
    sizes = [sampler.sample_patch_size() for _ in range(10)]
    logger.debug(f"Sampled patch sizes: {sizes}")

    # Test memory efficient trainer
    model = nn.Conv2d(1, 1, 3, padding=1)
    trainer = create_memory_efficient_trainer(model)

    hr_batch = torch.rand(2, 1, 256, 256)
    lr_batch = torch.rand(2, 1, 64, 64)

    metrics = trainer.train_step(hr_batch, lr_batch)
    logger.debug(f"Training metrics: {metrics}")

    logger.debug("Patch-wise processing test completed successfully!")
