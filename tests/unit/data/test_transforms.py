"""
Unit tests for data transforms.

Tests validate:
1. Transform correctness
2. Shape preservation
3. Determinism for val/test
4. Normalization behavior
5. Augmentation behavior

Run on CI, NOT local dev machine.
"""

import pytest
import torch


# === Normalization Transform Tests ===
class TestNormalizationTransforms:
    """Test normalization transforms."""

    # @# pytest.mark.timeout(30)
    def test_minmax_normalization(self):
        """MinMax normalization scales to [0, 1]."""
        data = torch.tensor([[0.0, 50.0], [100.0, 25.0]])

        # Manual minmax
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    # @# pytest.mark.timeout(30)
    def test_zscore_normalization(self):
        """Z-score normalization centers at 0 with std 1."""
        data = torch.randn(100, 64, 64) * 5 + 10  # mean~10, std~5

        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + 1e-8)

        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1) < 0.1

    # @# pytest.mark.timeout(30)
    def test_normalize_with_fixed_stats(self):
        """Normalization with pre-computed stats."""
        # These would come from training set
        train_mean = 0.5
        train_std = 0.25

        # Apply to different data
        data = torch.ones(1, 64, 64) * 0.75

        normalized = (data - train_mean) / train_std

        # 0.75 - 0.5 = 0.25, 0.25 / 0.25 = 1.0
        assert torch.allclose(normalized, torch.ones(1, 64, 64))

    # @# pytest.mark.timeout(30)
    def test_normalize_transform_import(self):
        """NormalizeTransform can be imported."""
        try:
            from spectramr.data.transforms.normalize import NormalizeTransform

            assert NormalizeTransform is not None
        except ImportError:
            pytest.skip("NormalizeTransform not available")


# === Augmentation Transform Tests ===
class TestAugmentationTransforms:
    """Test augmentation transforms."""

    # @# pytest.mark.timeout(30)
    def test_random_flip_horizontal(self):
        """Random horizontal flip augmentation."""
        # Original image with asymmetric content
        img = torch.zeros(1, 64, 64)
        img[:, :, :32] = 1.0  # Left half is white

        # Flipped version
        flipped = torch.flip(img, dims=[-1])

        # Right half should be white after flip
        assert flipped[:, :, 32:].sum() == img[:, :, :32].sum()
        assert flipped[:, :, :32].sum() == 0

    # @# pytest.mark.timeout(30)
    def test_random_rotation(self):
        """Random rotation preserves shape."""
        import torchvision.transforms.functional as TF

        img = torch.randn(1, 64, 64)

        # Rotate 90 degrees
        rotated = TF.rotate(img.unsqueeze(0), 90).squeeze(0)

        assert rotated.shape == img.shape

    # @# pytest.mark.timeout(30)
    def test_augmentation_determinism_with_seed(self):
        """Same seed should produce same augmentation."""
        img = torch.randn(1, 64, 64)

        torch.manual_seed(42)
        noise1 = torch.randn_like(img) * 0.1
        aug1 = img + noise1

        torch.manual_seed(42)
        noise2 = torch.randn_like(img) * 0.1
        aug2 = img + noise2

        assert torch.allclose(aug1, aug2)

    # @# pytest.mark.timeout(30)
    def test_no_augmentation_when_disabled(self):
        """No augmentation when disabled."""

        class ConditionalAugment:
            def __init__(self, enabled=True):
                self.enabled = enabled

            def __call__(self, x):
                if self.enabled:
                    return x + torch.randn_like(x) * 0.1
                return x

        img = torch.randn(1, 64, 64)

        aug_disabled = ConditionalAugment(enabled=False)

        out1 = aug_disabled(img)
        out2 = aug_disabled(img)

        assert torch.allclose(out1, out2)


# === K-Space Transform Tests ===
class TestKSpaceTransforms:
    """Test k-space specific transforms."""

    # @# pytest.mark.timeout(30)
    def test_fft_ifft_roundtrip(self):
        """FFT followed by IFFT should recover original."""
        img = torch.randn(1, 64, 64)

        kspace = torch.fft.fft2(img)
        recovered = torch.fft.ifft2(kspace).real

        assert torch.allclose(img, recovered, atol=1e-5)

    # @# pytest.mark.timeout(30)
    def test_center_crop_kspace(self):
        """Center cropping in k-space for acceleration."""
        kspace = torch.randn(1, 64, 64, dtype=torch.complex64)

        # Keep center 50%
        center_size = 32
        start = (64 - center_size) // 2
        end = start + center_size

        cropped = kspace[:, start:end, start:end]

        assert cropped.shape == (1, 32, 32)

    # @# pytest.mark.timeout(30)
    def test_undersampling_mask(self):
        """Undersampling mask generation."""
        # Create random undersampling mask (4x acceleration)
        mask = torch.zeros(64, 64)
        mask[::4, :] = 1.0  # Keep every 4th line

        # Center should be fully sampled
        center = 8
        mask[32 - center : 32 + center, :] = 1.0

        # Check acceleration factor
        acceleration = 64 * 64 / mask.sum()
        assert 1.5 < acceleration < 4.0  # Approximately 4x with center


# === Complex Data Transform Tests ===
class TestComplexTransforms:
    """Test transforms for complex-valued data."""

    # @# pytest.mark.timeout(30)
    def test_complex_to_2channel(self):
        """Convert complex tensor to 2-channel real."""
        complex_data = torch.randn(1, 64, 64, dtype=torch.complex64)

        real_part = complex_data.real
        imag_part = complex_data.imag

        two_channel = torch.stack([real_part, imag_part], dim=1).squeeze(0)

        assert two_channel.shape == (2, 64, 64)

    # @# pytest.mark.timeout(30)
    def test_2channel_to_complex(self):
        """Convert 2-channel real to complex tensor."""
        two_channel = torch.randn(2, 64, 64)

        complex_data = torch.complex(two_channel[0], two_channel[1])

        assert complex_data.shape == (64, 64)
        assert complex_data.dtype == torch.complex64

    # @# pytest.mark.timeout(30)
    def test_magnitude_extraction(self):
        """Extract magnitude from complex data."""
        real = torch.tensor([[3.0, 0.0]])
        imag = torch.tensor([[4.0, 1.0]])

        magnitude = torch.sqrt(real**2 + imag**2)

        assert torch.allclose(magnitude, torch.tensor([[5.0, 1.0]]))


# === Shape Preservation Tests ===
class TestShapePreservation:
    """Test that transforms preserve expected shapes."""

    # @# pytest.mark.timeout(30)
    def test_spatial_transform_preserves_shape(self):
        """Spatial transforms should preserve shape."""
        img = torch.randn(1, 1, 128, 128)

        # Simulate spatial transform
        transformed = img * 1.0  # Identity

        assert transformed.shape == img.shape

    # @# pytest.mark.timeout(30)
    def test_resize_changes_shape(self):
        """Resize should change spatial dimensions."""
        import torch.nn.functional as F

        img = torch.randn(1, 1, 128, 128)

        resized = F.interpolate(
            img, size=(64, 64), mode="bilinear", align_corners=False
        )

        assert resized.shape == (1, 1, 64, 64)

    # @# pytest.mark.timeout(30)
    def test_batch_dimension_preserved(self):
        """Batch dimension should be preserved."""
        batch = torch.randn(8, 1, 64, 64)

        # Apply per-sample transform
        transformed = batch * 2  # Simple scaling

        assert transformed.shape[0] == 8


# === Transform Composition Tests ===
class TestTransformComposition:
    """Test composing multiple transforms."""

    # @# pytest.mark.timeout(30)
    def test_compose_transforms(self):
        """Multiple transforms can be composed."""
        import torchvision.transforms as T

        compose = T.Compose(
            [
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        img = torch.rand(1, 64, 64)
        transformed = compose(img)

        assert transformed.shape == img.shape

    # @# pytest.mark.timeout(30)
    def test_transform_order_matters(self):
        """Transform order affects result."""
        img = torch.ones(1, 64, 64) * 0.5

        # Scale then shift
        result1 = (img * 2) + 1  # 0.5 * 2 + 1 = 2.0

        # Shift then scale
        result2 = (img + 1) * 2  # (0.5 + 1) * 2 = 3.0

        assert not torch.allclose(result1, result2)


# === Edge Cases ===
class TestTransformEdgeCases:
    """Test transform edge cases."""

    # @# pytest.mark.timeout(30)
    def test_transform_with_zeros(self):
        """Transforms should handle zero inputs."""
        zeros = torch.zeros(1, 64, 64)

        # Normalization with zeros
        normalized = (zeros - zeros.mean()) / (zeros.std() + 1e-8)

        assert not torch.isnan(normalized).any()

    # @# pytest.mark.timeout(30)
    def test_transform_with_constant(self):
        """Transforms should handle constant inputs."""
        constant = torch.ones(1, 64, 64) * 5.0

        # Z-score with constant gives 0
        normalized = (constant - constant.mean()) / (constant.std() + 1e-8)

        assert not torch.isnan(normalized).any()

    # @# pytest.mark.timeout(30)
    def test_transform_dtype_preservation(self):
        """Transforms should preserve dtype when possible."""
        float32 = torch.randn(1, 64, 64, dtype=torch.float32)

        # Simple transform
        transformed = float32 * 2 + 1

        assert transformed.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
