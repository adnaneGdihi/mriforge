"""Comprehensive unit tests for TRELLIS normalization layers.

Tests LayerNorm32, GroupNorm32, and ChannelLayerNorm32 classes with
numerical stability, dtype preservation, and gradient flow validation.
"""

import pytest
import torch
import torch.nn as nn

from spectramr.models.blocks.trellis_norm import ChannelLayerNorm32, GroupNorm32, LayerNorm32


class TestLayerNorm32:
    """Test LayerNorm32 class."""

    def test_initialization(self):
        """Test LayerNorm32 initialization."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape)

        assert isinstance(norm, nn.LayerNorm)
        assert norm.normalized_shape == torch.Size(normalized_shape)
        assert norm.elementwise_affine is True

    def test_initialization_with_options(self):
        """Test LayerNorm32 initialization with custom options."""
        normalized_shape = [32, 64]
        norm = LayerNorm32(normalized_shape, elementwise_affine=False, eps=1e-6)

        assert norm.normalized_shape == torch.Size(normalized_shape)
        assert norm.elementwise_affine is False
        assert norm.eps == 1e-6

    def test_forward_pass_float32_input(self):
        """Test forward pass with float32 input."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape)

        x = torch.randn(2, 10, 64)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.dtype == torch.float32

    def test_forward_pass_float16_input(self):
        """Test forward pass with float16 input (dtype preservation)."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape)

        x = torch.randn(2, 10, 64, dtype=torch.float16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.dtype == torch.float16

    def test_forward_pass_bfloat16_input(self):
        """Test forward pass with bfloat16 input."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape)

        x = torch.randn(2, 10, 64, dtype=torch.bfloat16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.dtype == torch.bfloat16

    def test_numerical_stability(self):
        """Test that LayerNorm32 provides numerical stability."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape)

        # Create input that might cause numerical issues in lower precision
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        x *= 1000  # Large scale

        output = norm(x)

        # Should not have NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Should be properly normalized (mean close to 0, std close to 1)
        mean = output.mean(dim=-1, keepdim=True)
        std = output.std(dim=-1, keepdim=True)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)

    def test_gradient_flow(self):
        """Test gradient flow through LayerNorm32."""
        normalized_shape = [32]
        norm = LayerNorm32(normalized_shape)

        x = torch.randn(2, 8, 32, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_no_affine_parameters(self):
        """Test LayerNorm32 without affine parameters."""
        normalized_shape = [64]
        norm = LayerNorm32(normalized_shape, elementwise_affine=False)

        x = torch.randn(2, 10, 64)
        output = norm(x)

        assert output.shape == x.shape
        # Should not have weight and bias parameters
        assert not hasattr(norm, "weight") or norm.weight is None
        assert not hasattr(norm, "bias") or norm.bias is None


class TestGroupNorm32:
    """Test GroupNorm32 class."""

    def test_initialization(self):
        """Test GroupNorm32 initialization."""
        num_groups = 8
        num_channels = 64
        norm = GroupNorm32(num_groups, num_channels)

        assert isinstance(norm, nn.GroupNorm)
        assert norm.num_groups == num_groups
        assert norm.num_channels == num_channels
        assert norm.affine is True

    def test_initialization_with_options(self):
        """Test GroupNorm32 initialization with custom options."""
        num_groups = 4
        num_channels = 32
        norm = GroupNorm32(num_groups, num_channels, affine=False, eps=1e-5)

        assert norm.num_groups == num_groups
        assert norm.num_channels == num_channels
        assert norm.affine is False
        assert norm.eps == 1e-5

    def test_forward_pass_2d_input(self):
        """Test forward pass with 2D input."""
        num_groups = 8
        num_channels = 64
        norm = GroupNorm32(num_groups, num_channels)

        x = torch.randn(2, 64, 32, 32)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input."""
        num_groups = 4
        num_channels = 32
        norm = GroupNorm32(num_groups, num_channels)

        x = torch.randn(2, 32, 8, 16, 16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_forward_pass_float16_input(self):
        """Test forward pass with float16 input."""
        num_groups = 8
        num_channels = 64
        norm = GroupNorm32(num_groups, num_channels)

        x = torch.randn(2, 64, 16, 16, dtype=torch.float16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == torch.float16

    def test_numerical_stability(self):
        """Test numerical stability of GroupNorm32."""
        num_groups = 4
        num_channels = 32
        norm = GroupNorm32(num_groups, num_channels)

        x = torch.randn(2, 32, 8, 8, dtype=torch.float16)
        x *= 100  # Large scale

        output = norm(x)

        # Should not have NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test gradient flow through GroupNorm32."""
        num_groups = 4
        num_channels = 32
        norm = GroupNorm32(num_groups, num_channels)

        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_invalid_num_groups(self):
        """Test error when num_groups doesn't divide num_channels."""
        with pytest.raises(
            ValueError, match=r"num_channels.*must be divisible by num_groups"
        ):
            GroupNorm32(7, 64)  # 64 not divisible by 7


class TestChannelLayerNorm32:
    """Test ChannelLayerNorm32 class."""

    def test_initialization(self):
        """Test ChannelLayerNorm32 initialization."""
        normalized_shape = [64]
        norm = ChannelLayerNorm32(normalized_shape)

        assert isinstance(norm, LayerNorm32)
        assert norm.normalized_shape == torch.Size(normalized_shape)

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input (batch, channels, spatial)."""
        normalized_shape = [64]
        norm = ChannelLayerNorm32(normalized_shape)

        x = torch.randn(2, 64, 32)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_forward_pass_4d_input(self):
        """Test forward pass with 4D input (batch, channels, height, width)."""
        normalized_shape = [128]
        norm = ChannelLayerNorm32(normalized_shape)

        x = torch.randn(2, 128, 16, 16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_forward_pass_5d_input(self):
        """Test forward pass with 5D input (batch, channels, depth, height, width)."""
        normalized_shape = [64]
        norm = ChannelLayerNorm32(normalized_shape)

        x = torch.randn(2, 64, 8, 16, 16)
        output = norm(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_channel_wise_normalization(self):
        """Test that normalization is applied across channels per spatial position."""
        normalized_shape = [32]
        norm = ChannelLayerNorm32(normalized_shape)

        # Create input with specific channel patterns
        batch_size, channels, height, width = 2, 32, 4, 4
        x = torch.randn(batch_size, channels, height, width)

        # Make channels have different statistics at each spatial position
        for c in range(channels):
            x[:, c, :, :] += c * 0.1

        output = norm(x)

        # Check that channel statistics are normalized per spatial position
        # For each spatial position, the mean and std across channels should be
        # normalized
        channel_mean = output.mean(dim=1, keepdim=True)  # Mean across channels
        channel_std = output.std(dim=1, keepdim=True)  # Std across channels

        assert torch.allclose(channel_mean, torch.zeros_like(channel_mean), atol=1e-2)
        assert torch.allclose(channel_std, torch.ones_like(channel_std), atol=1e-1)

    def test_dtype_preservation(self):
        """Test dtype preservation in ChannelLayerNorm32."""
        normalized_shape = [64]
        norm = ChannelLayerNorm32(normalized_shape)

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            x = torch.randn(2, 64, 8, 8, dtype=dtype)
            output = norm(x)

            assert output.dtype == dtype

    def test_gradient_flow(self):
        """Test gradient flow through ChannelLayerNorm32."""
        normalized_shape = [32]
        norm = ChannelLayerNorm32(normalized_shape)

        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_numerical_stability(self):
        """Test numerical stability of ChannelLayerNorm32."""
        normalized_shape = [64]
        norm = ChannelLayerNorm32(normalized_shape)

        x = torch.randn(2, 64, 4, 4, dtype=torch.float16)
        x *= 1000  # Large scale

        output = norm(x)

        # Should not have NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTrellisNormIntegration:
    """Test integration between TRELLIS normalization components."""

    def test_layer_norm_compatibility(self):
        """Test that LayerNorm32 is compatible with standard LayerNorm interface."""
        normalized_shape = [64]
        norm32 = LayerNorm32(normalized_shape)
        norm_standard = nn.LayerNorm(normalized_shape)

        x = torch.randn(2, 10, 64)

        out32 = norm32(x)
        out_standard = norm_standard(x.float()).to(x.dtype)

        # Results should be very close (allowing for float32 computation differences)
        assert torch.allclose(out32, out_standard, rtol=1e-4)

    def test_group_norm_compatibility(self):
        """Test that GroupNorm32 is compatible with standard GroupNorm interface."""
        num_groups = 8
        num_channels = 64
        norm32 = GroupNorm32(num_groups, num_channels)
        norm_standard = nn.GroupNorm(num_groups, num_channels)

        x = torch.randn(2, 64, 16, 16)

        out32 = norm32(x)
        out_standard = norm_standard(x.float()).to(x.dtype)

        # Results should be very close
        assert torch.allclose(out32, out_standard, rtol=1e-4)

    def test_channel_norm_uniqueness(self):
        """Test that ChannelLayerNorm32 behaves differently from standard LayerNorm."""
        normalized_shape = [64]
        channel_norm = ChannelLayerNorm32(normalized_shape)
        standard_norm = LayerNorm32(normalized_shape)

        # Create input where channel-wise vs. feature-wise normalization matters
        x = torch.randn(2, 64, 8, 8)

        out_channel = channel_norm(x)
        out_standard = standard_norm(x.view(2, 64, -1)).view(2, 64, 8, 8)

        # Results should be different
        assert not torch.allclose(out_channel, out_standard, atol=1e-3)


class TestTrellisNormEdgeCases:
    """Test edge cases for TRELLIS normalization components."""

    def test_single_channel_normalization(self):
        """Test normalization with single channel."""
        norm = ChannelLayerNorm32([1])

        x = torch.randn(2, 1, 8, 8)
        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_large_batch_sizes(self):
        """Test with large batch sizes."""
        norm = LayerNorm32([64])

        x = torch.randn(64, 100, 64)
        output = norm(x)

        assert output.shape == x.shape

    def test_very_small_features(self):
        """Test with very small feature dimensions."""
        norm = LayerNorm32([2])

        x = torch.randn(2, 10, 2)
        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow_with_mixed_precision(self):
        """Test gradient flow with mixed precision."""
        norm = GroupNorm32(4, 32)

        x = torch.randn(2, 32, 8, 8, dtype=torch.float16, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.dtype == torch.float16


if __name__ == "__main__":
    pytest.main([__file__])
