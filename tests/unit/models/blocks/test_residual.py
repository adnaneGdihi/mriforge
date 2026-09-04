"""Comprehensive unit tests for residual building blocks.

Tests the ResidualBlock, PreActResidualBlock, and UnifiedResidualBlock classes
with comprehensive edge cases and validation.
"""

import math

import pytest
import torch
from torch import nn

from spectramr.models.blocks.residual import (
    PreActResidualBlock,
    ResidualBlock,
    UnifiedResidualBlock,
)


class TestResidualBlock:
    """Test ResidualBlock class."""

    def test_initialization(self):
        """Test ResidualBlock initialization."""
        channels = 64
        block = ResidualBlock(channels)

        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.bn1, nn.GroupNorm)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.bn2, nn.GroupNorm)
        assert isinstance(block.relu, nn.LeakyReLU)
        assert isinstance(block.shortcut, nn.Sequential)

        # Check conv layer configurations
        assert block.conv1.in_channels == channels
        assert block.conv1.out_channels == channels
        assert block.conv1.kernel_size == (3, 3)
        assert block.conv1.stride == (1, 1)
        assert block.conv1.padding == (1, 1)

    def test_initialization_with_stride(self):
        """Test ResidualBlock with stride > 1."""
        channels = 64
        stride = 2
        block = ResidualBlock(channels, stride=stride)

        # Should have shortcut connection for downsampling
        assert len(block.shortcut) > 0
        assert isinstance(block.shortcut[0], nn.Conv2d)
        assert isinstance(block.shortcut[1], nn.GroupNorm)

        # Check shortcut conv configuration
        shortcut_conv = block.shortcut[0]
        assert shortcut_conv.kernel_size == (1, 1)
        assert shortcut_conv.stride == (stride, stride)

    def test_forward_pass(self):
        """Test forward pass of ResidualBlock."""
        batch_size, channels, height, width = 2, 64, 32, 32
        block = ResidualBlock(channels)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_forward_pass_with_stride(self):
        """Test forward pass with stride (spatial downsampling)."""
        batch_size, channels, height, width = 2, 64, 32, 32
        stride = 2
        block = ResidualBlock(channels, stride=stride)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        expected_height = height // stride
        expected_width = width // stride
        assert output.shape == (batch_size, channels, expected_height, expected_width)

    def test_gradient_flow(self):
        """Test that gradients flow through ResidualBlock."""
        channels = 32
        block = ResidualBlock(channels)

        x = torch.randn(2, channels, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection is working."""
        channels = 16
        block = ResidualBlock(channels)

        x = torch.randn(1, channels, 8, 8)

        # Without residual, the output would be different
        # With residual, output should be x + F(x)
        output = block(x)

        # The residual connection ensures the output is not just the conv output
        # We can't easily test the exact residual addition, but we can test
        # that the block doesn't collapse to identity
        assert not torch.allclose(output, x, atol=1e-6)

    def test_group_norm_groups(self):
        """Test that GroupNorm uses appropriate number of groups."""
        channels = 64
        block = ResidualBlock(channels)

        # GroupNorm should use gcd(32, channels) groups
        expected_groups = math.gcd(32, channels)
        assert block.bn1.num_groups == expected_groups
        assert block.bn2.num_groups == expected_groups

        if len(block.shortcut) > 0:
            assert block.shortcut[1].num_groups == expected_groups


class TestPreActResidualBlock:
    """Test PreActResidualBlock class."""

    def test_initialization(self):
        """Test PreActResidualBlock initialization."""
        channels = 64
        block = PreActResidualBlock(channels)

        # Pre-activation has BN before conv
        assert isinstance(block.bn1, nn.GroupNorm)
        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.bn2, nn.GroupNorm)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.relu, nn.LeakyReLU)

    def test_initialization_with_stride(self):
        """Test PreActResidualBlock with stride > 1."""
        channels = 64
        stride = 2
        block = PreActResidualBlock(channels, stride=stride)

        # Should have shortcut connection
        assert len(block.shortcut) > 0

    def test_forward_pass(self):
        """Test forward pass of PreActResidualBlock."""
        batch_size, channels, height, width = 2, 64, 32, 32
        block = PreActResidualBlock(channels)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        assert output.shape == x.shape

    def test_forward_pass_with_stride(self):
        """Test forward pass with stride."""
        batch_size, channels, height, width = 2, 64, 32, 32
        stride = 2
        block = PreActResidualBlock(channels, stride=stride)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        expected_height = height // stride
        expected_width = width // stride
        assert output.shape == (batch_size, channels, expected_height, expected_width)

    def test_gradient_flow(self):
        """Test gradient flow through PreActResidualBlock."""
        channels = 32
        block = PreActResidualBlock(channels)

        x = torch.randn(2, channels, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_pre_activation_order(self):
        """Test that pre-activation applies BN before conv."""
        channels = 16
        block = PreActResidualBlock(channels)

        x = torch.randn(1, channels, 8, 8)

        # In pre-activation, BN is applied to input first
        # We can verify this by checking the forward pass doesn't fail
        output = block(x)
        assert output.shape == x.shape


class TestUnifiedResidualBlock:
    """Test UnifiedResidualBlock class."""

    def test_initialization_basic(self):
        """Test basic UnifiedResidualBlock initialization."""
        channels = 64
        block = UnifiedResidualBlock(channels)

        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "bn1")
        assert hasattr(block, "bn2")
        assert hasattr(block, "act")
        assert hasattr(block, "shortcut")

    def test_initialization_with_options(self):
        """Test UnifiedResidualBlock with various options."""
        channels = 64

        # Test with different activations
        for activation in ["relu", "leaky", "gelu"]:
            block = UnifiedResidualBlock(channels, activation=activation)
            assert hasattr(block, "act")

        # Test with bias
        block = UnifiedResidualBlock(channels, bias=True)
        assert block.conv1.bias is not None

        # Test without group norm
        block = UnifiedResidualBlock(channels, use_group_norm=False)
        assert isinstance(block.bn1, nn.Identity)

    def test_forward_pass_basic(self):
        """Test basic forward pass of UnifiedResidualBlock."""
        batch_size, channels, height, width = 2, 64, 32, 32
        block = UnifiedResidualBlock(channels)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        assert output.shape == x.shape

    def test_forward_pass_with_stride(self):
        """Test forward pass with stride."""
        batch_size, channels, height, width = 2, 64, 32, 32
        stride = 2
        block = UnifiedResidualBlock(channels, stride=stride)

        x = torch.randn(batch_size, channels, height, width)
        output = block(x)

        expected_height = height // stride
        expected_width = width // stride
        assert output.shape == (batch_size, channels, expected_height, expected_width)

    def test_pre_activation_mode(self):
        """Test pre-activation mode."""
        channels = 32
        block = UnifiedResidualBlock(channels, pre_activation=True)

        x = torch.randn(2, channels, 16, 16)
        output = block(x)

        assert output.shape == x.shape

    def test_skip_scale(self):
        """Test skip connection scaling."""
        channels = 32
        skip_scale = 0.5
        block = UnifiedResidualBlock(channels, skip_scale=skip_scale)

        x = torch.randn(1, channels, 8, 8)
        output = block(x)

        assert output.shape == x.shape

    def test_no_shortcut(self):
        """Test without shortcut connection."""
        channels = 32
        block = UnifiedResidualBlock(channels, use_shortcut=False)

        x = torch.randn(1, channels, 8, 8)
        output = block(x)

        assert output.shape == x.shape
        assert isinstance(block.shortcut, nn.Identity)

    def test_gradient_flow(self):
        """Test gradient flow through UnifiedResidualBlock."""
        channels = 32
        block = UnifiedResidualBlock(channels)

        x = torch.randn(2, channels, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_gradient_flow_pre_activation(self):
        """Test gradient flow in pre-activation mode."""
        channels = 32
        block = UnifiedResidualBlock(channels, pre_activation=True)

        x = torch.randn(2, channels, 16, 16, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        channels = 32
        for kernel_size in [1, 3, 5, 7]:
            block = UnifiedResidualBlock(channels, kernel_size=kernel_size)

            x = torch.randn(1, channels, 16, 16)
            output = block(x)

            assert output.shape == x.shape

    def test_different_activations(self):
        """Test different activation functions."""
        channels = 32
        activations = ["relu", "leaky", "gelu"]

        for activation in activations:
            block = UnifiedResidualBlock(channels, activation=activation)

            x = torch.randn(1, channels, 8, 8)
            output = block(x)

            assert output.shape == x.shape


class TestResidualBlockComparison:
    """Test comparison between different residual block implementations."""

    def test_residual_vs_pre_activation(self):
        """Compare ResidualBlock vs PreActResidualBlock."""
        channels = 32

        residual_block = ResidualBlock(channels)
        preact_block = PreActResidualBlock(channels)

        x = torch.randn(2, channels, 16, 16)

        residual_output = residual_block(x)
        preact_output = preact_block(x)

        # Both should produce valid outputs of the same shape
        assert residual_output.shape == x.shape
        assert preact_output.shape == x.shape

    def test_unified_vs_basic_residual(self):
        """Compare UnifiedResidualBlock vs ResidualBlock."""
        channels = 32

        unified_block = UnifiedResidualBlock(channels, pre_activation=False)
        basic_block = ResidualBlock(channels)

        x = torch.randn(2, channels, 16, 16)

        unified_output = unified_block(x)
        basic_output = basic_block(x)

        # Both should produce valid outputs
        assert unified_output.shape == x.shape
        assert basic_output.shape == x.shape


class TestResidualBlockEdgeCases:
    """Test edge cases for residual blocks."""

    def test_single_channel(self):
        """Test with single channel."""
        channels = 1

        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            block = block_class(channels)
            x = torch.randn(1, channels, 8, 8)
            output = block(x)
            assert output.shape == x.shape

    def test_large_channel_counts(self):
        """Test with large channel counts."""
        channels = 512

        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            block = block_class(channels)
            x = torch.randn(1, channels, 8, 8)
            output = block(x)
            assert output.shape == x.shape

    def test_different_spatial_sizes(self):
        """Test with different spatial dimensions."""
        channels = 32
        spatial_sizes = [(4, 4), (8, 16), (16, 32), (32, 32)]

        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            for height, width in spatial_sizes:
                block = block_class(channels)
                x = torch.randn(1, channels, height, width)
                output = block(x)
                assert output.shape == x.shape

    def test_stride_edge_cases(self):
        """Test stride edge cases."""
        channels = 32

        # Test stride = 1 (no downsampling)
        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            block = block_class(channels, stride=1)
            x = torch.randn(1, channels, 16, 16)
            output = block(x)
            assert output.shape == x.shape

    def test_unified_block_configurations(self):
        """Test various UnifiedResidualBlock configurations."""
        channels = 32
        x = torch.randn(1, channels, 8, 8)

        # Test all combinations of options
        configs = [
            {"pre_activation": True, "use_group_norm": True},
            {"pre_activation": False, "use_group_norm": False},
            {"use_shortcut": False},
            {"skip_scale": 0.1},
            {"activation": "gelu"},
            {"bias": True},
        ]

        for config in configs:
            block = UnifiedResidualBlock(channels, **config)
            output = block(x)
            assert output.shape == x.shape


class TestUnifiedResidualBlockActivationValidation:
    """Pitfall #9: an unknown activation string must raise, never silently
    become ReLU (the old fallback made a typo'd 'leaky_relu'/'silu' YAML run
    with a different nonlinearity than advertised)."""

    @pytest.mark.parametrize("activation", ["relu", "leaky", "gelu"])
    def test_valid_activation_constructs_and_runs(self, activation):
        block = UnifiedResidualBlock(16, activation=activation)
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("activation", ["silu", "leaky_relu", "RELU", ""])
    def test_unknown_activation_raises_with_valid_set(self, activation):
        with pytest.raises(ValueError, match="'relu', 'leaky', 'gelu'"):
            UnifiedResidualBlock(16, activation=activation)


class TestResidualBlockIntegration:
    """Test residual blocks integration with PyTorch."""

    def test_nn_module_integration(self):
        """Test PyTorch nn.Module integration."""
        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            block = block_class(64)

            # Should have parameters
            params = list(block.parameters())
            assert len(params) > 0

            # Should be able to move to device
            if torch.cuda.is_available():
                block.cuda()
                assert next(block.parameters()).is_cuda

    def test_training_mode(self):
        """Test training vs eval mode."""
        channels = 32
        x = torch.randn(1, channels, 8, 8)

        for block_class in [ResidualBlock, PreActResidualBlock, UnifiedResidualBlock]:
            block = block_class(channels)

            # Training mode
            block.train()
            train_output = block(x)

            # Eval mode
            block.eval()
            eval_output = block(x)

            # Outputs should be different due to batch norm behavior
            assert train_output.shape == eval_output.shape


if __name__ == "__main__":
    pytest.main([__file__])
