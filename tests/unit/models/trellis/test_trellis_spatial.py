"""Tests for TRELLIS spatial utilities.

Tests patchify and unpatchify functions for various dimensionalities and edge cases.
"""

import pytest
import torch

from mriforge.models.blocks.trellis_spatial import patchify, unpatchify


class TestPatchify:
    """Test patchify function."""

    def test_patchify_2d_basic(self):
        """Test basic 2D patchify operation."""
        batch_size, channels, height, width = 2, 3, 8, 8
        patch_size = 4
        x = torch.randn(batch_size, channels, height, width)

        result = patchify(x, patch_size)

        # Expected shape: (batch, num_patches_h, num_patches_w, channels * patch_size * patch_size)
        expected_shape = (
            batch_size,
            height // patch_size,
            width // patch_size,
            channels * (patch_size**2),
        )
        assert result.shape == expected_shape
        assert result.dtype == x.dtype

    def test_patchify_3d_basic(self):
        """Test basic 3D patchify operation."""
        batch_size, channels, depth, height, width = 2, 3, 4, 8, 8
        patch_size = 2
        x = torch.randn(batch_size, channels, depth, height, width)

        result = patchify(x, patch_size)

        # Expected shape: (batch, num_patches_d, num_patches_h, num_patches_w, channels * patch_size^3)
        expected_shape = (
            batch_size,
            depth // patch_size,
            height // patch_size,
            width // patch_size,
            channels * (patch_size**3),
        )
        assert result.shape == expected_shape

    def test_patchify_single_patch(self):
        """Test patchify with single patch per dimension."""
        batch_size, channels, height, width = 1, 2, 4, 4
        patch_size = 4
        x = torch.randn(batch_size, channels, height, width)

        result = patchify(x, patch_size)

        expected_shape = (batch_size, 1, 1, channels * (patch_size**2))
        assert result.shape == expected_shape

    def test_patchify_different_patch_sizes(self):
        """Test patchify with different patch sizes."""
        batch_size, channels, height, width = 1, 1, 6, 6
        x = torch.randn(batch_size, channels, height, width)

        # Test patch_size = 2
        result2 = patchify(x, 2)
        expected_shape2 = (batch_size, 3, 3, channels * 4)
        assert result2.shape == expected_shape2

        # Test patch_size = 3
        result3 = patchify(x, 3)
        expected_shape3 = (batch_size, 2, 2, channels * 9)
        assert result3.shape == expected_shape3

    def test_patchify_non_divisible_height(self):
        """Test patchify raises error for non-divisible height."""
        x = torch.randn(1, 1, 7, 8)  # Height 7 not divisible by 4
        with pytest.raises(
            ValueError, match="Dimension 2 must be divisible by patch_size"
        ):
            patchify(x, 4)

    def test_patchify_non_divisible_width(self):
        """Test patchify raises error for non-divisible width."""
        x = torch.randn(1, 1, 8, 7)  # Width 7 not divisible by 4
        with pytest.raises(
            ValueError, match="Dimension 3 must be divisible by patch_size"
        ):
            patchify(x, 4)

    def test_patchify_3d_non_divisible_depth(self):
        """Test patchify raises error for non-divisible depth in 3D."""
        x = torch.randn(1, 1, 3, 4, 4)  # Depth 3 not divisible by 2
        with pytest.raises(
            ValueError, match="Dimension 2 must be divisible by patch_size"
        ):
            patchify(x, 2)

    def test_patchify_preserve_dtype(self):
        """Test patchify preserves tensor dtype."""
        x = torch.randn(1, 1, 4, 4).to(torch.float32)
        result = patchify(x, 2)
        assert result.dtype == torch.float32

        x_double = torch.randn(1, 1, 4, 4).to(torch.float64)
        result_double = patchify(x_double, 2)
        assert result_double.dtype == torch.float64


class TestUnpatchify:
    """Test unpatchify function."""

    def test_unpatchify_2d_basic(self):
        """Test basic 2D unpatchify operation."""
        batch_size, channels, num_patches_h, num_patches_w = 2, 3, 2, 2
        patch_size = 4
        # Create patched tensor: (batch, num_patches_h, num_patches_w, channels * patch_size^2)
        x = torch.randn(
            batch_size, num_patches_h, num_patches_w, channels * (patch_size**2)
        )

        result = unpatchify(x, patch_size)

        expected_shape = (
            batch_size,
            channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        assert result.shape == expected_shape
        assert result.dtype == x.dtype

    def test_unpatchify_3d_basic(self):
        """Test basic 3D unpatchify operation."""
        batch_size, channels, num_patches_d, num_patches_h, num_patches_w = (
            2,
            3,
            2,
            2,
            2,
        )
        patch_size = 2
        # Create patched tensor: (batch, num_patches_d, num_patches_h, num_patches_w, channels * patch_size^3)
        x = torch.randn(
            batch_size,
            num_patches_d,
            num_patches_h,
            num_patches_w,
            channels * (patch_size**3),
        )

        result = unpatchify(x, patch_size)

        expected_shape = (
            batch_size,
            channels,
            num_patches_d * patch_size,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        assert result.shape == expected_shape

    def test_unpatchify_single_patch(self):
        """Test unpatchify with single patch per dimension."""
        batch_size, channels = 1, 2
        patch_size = 4
        # Single patch: (batch, 1, 1, channels * patch_size^2)
        x = torch.randn(batch_size, 1, 1, channels * (patch_size**2))

        result = unpatchify(x, patch_size)

        expected_shape = (batch_size, channels, patch_size, patch_size)
        assert result.shape == expected_shape

    def test_unpatchify_different_patch_sizes(self):
        """Test unpatchify with different patch sizes."""
        batch_size, channels = 1, 1

        # Test patch_size = 2: (batch, 3, 3, channels * 4) -> (batch, channels, 6, 6)
        x2 = torch.randn(batch_size, 3, 3, channels * 4)
        result2 = unpatchify(x2, 2)
        assert result2.shape == (batch_size, channels, 6, 6)

        # Test patch_size = 3: (batch, 2, 2, channels * 9) -> (batch, channels, 6, 6)
        x3 = torch.randn(batch_size, 2, 2, channels * 9)
        result3 = unpatchify(x3, 3)
        assert result3.shape == (batch_size, channels, 6, 6)

    def test_unpatchify_invalid_channel_dimension(self):
        """Test unpatchify raises error for invalid channel dimension."""
        # 2D case: channels * patch_size^2 should divide the last dimension
        x = torch.randn(1, 2, 2, 7)  # 7 is not divisible by 4 (1 * 2^2)
        with pytest.raises(ValueError, match="Last dimension must be divisible"):
            unpatchify(x, 2)

    def test_unpatchify_3d_invalid_channel_dimension(self):
        """Test unpatchify raises error for invalid channel dimension in 3D."""
        # 3D case: channels * patch_size^3 should divide the last dimension
        x = torch.randn(1, 2, 2, 2, 7)  # 7 is not divisible by 8 (1 * 2^3)
        with pytest.raises(ValueError, match="Last dimension must be divisible"):
            unpatchify(x, 2)

    def test_unpatchify_preserve_dtype(self):
        """Test unpatchify preserves tensor dtype."""
        x = torch.randn(1, 2, 2, 4).to(torch.float32)
        result = unpatchify(x, 2)
        assert result.dtype == torch.float32

        x_double = torch.randn(1, 2, 2, 4).to(torch.float64)
        result_double = unpatchify(x_double, 2)
        assert result_double.dtype == torch.float64


class TestPatchifyUnpatchifyRoundTrip:
    """Test round-trip consistency between patchify and unpatchify."""

    def test_round_trip_2d(self):
        """Test that patchify -> unpatchify gives original tensor."""
        batch_size, channels, height, width = 2, 3, 8, 8
        patch_size = 4
        x = torch.randn(batch_size, channels, height, width)

        # Round trip
        patched = patchify(x, patch_size)
        reconstructed = unpatchify(patched, patch_size)

        assert torch.allclose(x, reconstructed, rtol=1e-5)
        assert x.shape == reconstructed.shape

    def test_round_trip_3d(self):
        """Test that patchify -> unpatchify gives original tensor for 3D."""
        batch_size, channels, depth, height, width = 1, 2, 4, 6, 6
        patch_size = 2
        x = torch.randn(batch_size, channels, depth, height, width)

        # Round trip
        patched = patchify(x, patch_size)
        reconstructed = unpatchify(patched, patch_size)

        assert torch.allclose(x, reconstructed, rtol=1e-5)
        assert x.shape == reconstructed.shape

    def test_round_trip_different_patch_sizes(self):
        """Test round-trip with different patch sizes."""
        x = torch.randn(1, 1, 6, 6)

        for patch_size in [2, 3]:
            patched = patchify(x, patch_size)
            reconstructed = unpatchify(patched, patch_size)
            assert torch.allclose(x, reconstructed, rtol=1e-5)

    def test_round_trip_gradient_flow(self):
        """Test that gradients flow through round-trip operations."""
        x = torch.randn(1, 1, 4, 4, requires_grad=True)
        patch_size = 2

        # Forward pass
        patched = patchify(x, patch_size)
        reconstructed = unpatchify(patched, patch_size)

        # Backward pass
        loss = reconstructed.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSpatialEdgeCases:
    """Test edge cases for spatial operations."""

    def test_patchify_minimum_dimensions(self):
        """Test patchify with minimum valid dimensions."""
        # 2D: minimum 1x1 spatial dimensions with patch_size=1
        x = torch.randn(1, 1, 1, 1)
        result = patchify(x, 1)
        expected_shape = (1, 1, 1, 1)
        assert result.shape == expected_shape

    def test_unpatchify_minimum_dimensions(self):
        """Test unpatchify with minimum valid dimensions."""
        # 2D: minimum 1x1 patches with patch_size=1
        x = torch.randn(1, 1, 1, 1)
        result = unpatchify(x, 1)
        expected_shape = (1, 1, 1, 1)
        assert result.shape == expected_shape

    def test_patchify_large_batch(self):
        """Test patchify with large batch size."""
        batch_size, channels, height, width = 16, 3, 8, 8
        patch_size = 4
        x = torch.randn(batch_size, channels, height, width)

        result = patchify(x, patch_size)
        expected_shape = (batch_size, 2, 2, channels * 16)
        assert result.shape == expected_shape

    def test_unpatchify_large_batch(self):
        """Test unpatchify with large batch size."""
        batch_size, channels, num_patches_h, num_patches_w = 16, 3, 2, 2
        patch_size = 4
        x = torch.randn(batch_size, num_patches_h, num_patches_w, channels * 16)

        result = unpatchify(x, patch_size)
        expected_shape = (batch_size, channels, 8, 8)
        assert result.shape == expected_shape

    def test_patchify_different_channel_counts(self):
        """Test patchify with different channel counts."""
        for channels in [1, 3, 64]:
            x = torch.randn(1, channels, 4, 4)
            result = patchify(x, 2)
            expected_shape = (1, 2, 2, channels * 4)
            assert result.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__])
