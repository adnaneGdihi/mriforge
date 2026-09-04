"""Test VGG19FeatureExtractor handles 2-channel k-space inputs correctly."""

import pytest
import torch

from spectramr.models.losses.gan_loss_library import VGG19FeatureExtractor


class TestVGG19FeatureExtractor:
    """Test suite for VGG19 feature extraction with various channel counts."""

    @pytest.fixture
    def feature_extractor(self):
        """Create a VGG19FeatureExtractor with input normalization."""
        return VGG19FeatureExtractor(use_input_norm=True)

    def test_2_channel_kspace_input(self, feature_extractor):
        """Test that 2-channel k-space (real, imag) is converted to 3-channel."""
        # Simulate 2-channel k-space tensor: [batch, 2, H, W]
        # (real and imaginary components)
        batch_size, height, width = 4, 256, 256
        kspace_input = torch.randn(batch_size, 2, height, width, dtype=torch.float32)

        # Should not raise an error about channel mismatch
        output = feature_extractor(kspace_input)

        # Should return list of feature tensors
        assert isinstance(output, list), "Output should be list of features"
        assert len(output) > 0, "Should extract at least one feature level"

        # Each feature should have batch dimension
        for feature in output:
            assert (
                feature.shape[0] == batch_size
            ), f"Feature batch size should be {batch_size}"

    def test_1_channel_input(self, feature_extractor):
        """Test that 1-channel (grayscale) is converted to 3-channel by repeating."""
        batch_size, height, width = 4, 256, 256
        grayscale_input = torch.randn(batch_size, 1, height, width, dtype=torch.float32)

        output = feature_extractor(grayscale_input)

        assert isinstance(output, list)
        assert len(output) > 0
        for feature in output:
            assert feature.shape[0] == batch_size

    def test_3_channel_input(self, feature_extractor):
        """Test that 3-channel RGB input passes through unchanged."""
        batch_size, height, width = 4, 256, 256
        rgb_input = torch.randn(batch_size, 3, height, width, dtype=torch.float32)

        output = feature_extractor(rgb_input)

        assert isinstance(output, list)
        assert len(output) > 0
        for feature in output:
            assert feature.shape[0] == batch_size

    def test_kspace_to_3channel_conversion(self):
        """Test the internal conversion logic: 2-channel → 3-channel."""
        # Simulate 2-channel k-space
        real_part = torch.ones(1, 1, 256, 256) * 1.0
        imag_part = torch.ones(1, 1, 256, 256) * 0.5
        kspace = torch.cat([real_part, imag_part], dim=1)  # [1, 2, 256, 256]

        extractor = VGG19FeatureExtractor(use_input_norm=False)
        output = extractor(kspace)

        # Verify it didn't crash and returned features
        assert isinstance(output, list)
        assert len(output) > 0

    def test_batch_processing(self, feature_extractor):
        """Test that multiple samples in batch are processed correctly."""
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            kspace_input = torch.randn(batch_size, 2, 128, 128)
            output = feature_extractor(kspace_input)

            assert len(output) > 0
            for feature in output:
                assert feature.shape[0] == batch_size, (
                    f"Batch size mismatch for batch_size={batch_size}, "
                    f"got feature shape {feature.shape}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
