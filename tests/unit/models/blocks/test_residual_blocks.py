"""Unit tests for residual blocks.

Tests residual connection patterns including basic residual blocks,
pre-activation variants, and unified configurable residual blocks.
"""

import pytest  # noqa: F401
import torch

from mriforge.models.blocks.residual import (
    PreActResidualBlock,
    ResidualBlock,
    UnifiedResidualBlock,
)


class TestResidualBlock:
    """Test ResidualBlock."""

    def test_init_default_parameters(self):
        """Test ResidualBlock initialization with defaults."""
        block = ResidualBlock(channels=64)

        assert isinstance(block.conv1, torch.nn.Conv2d)
        assert isinstance(block.conv2, torch.nn.Conv2d)
        assert isinstance(block.bn1, torch.nn.GroupNorm)
        assert isinstance(block.bn2, torch.nn.GroupNorm)
        assert isinstance(block.relu, torch.nn.LeakyReLU)
        assert block.conv1.kernel_size == (3, 3)
        assert block.conv1.stride == (1, 1)

    def test_init_custom_parameters(self):
        """Test ResidualBlock initialization with custom parameters."""
        block = ResidualBlock(channels=128, kernel_size=5, stride=2, padding=2)

        assert block.conv1.kernel_size == (5, 5)
        assert block.conv1.stride == (2, 2)
        assert block.conv1.padding == (2, 2)
        # Second conv should have stride=1
        assert block.conv2.stride == (1, 1)

    def test_init_stride_creates_shortcut(self):
        """Test that stride > 1 creates a shortcut convolution."""
        block_stride1 = ResidualBlock(channels=64, stride=1)
        block_stride2 = ResidualBlock(channels=64, stride=2)

        # stride=1 should have identity shortcut
        assert len(block_stride1.shortcut) == 0

        # stride=2 should have conv shortcut
        assert len(block_stride2.shortcut) == 2  # Conv2d + GroupNorm

    def test_forward_pass_identity_shortcut(self):
        """Test forward pass with identity shortcut (stride=1)."""
        block = ResidualBlock(channels=64, stride=1)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_projection_shortcut(self):
        """Test forward pass with projection shortcut (stride=2)."""
        block = ResidualBlock(channels=64, stride=2)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        # Should downsample spatially
        assert output.shape == (2, 64, 8, 8)
        assert isinstance(output, torch.Tensor)


class TestPreActResidualBlock:
    """Test PreActResidualBlock."""

    def test_init_default_parameters(self):
        """Test PreActResidualBlock initialization with defaults."""
        block = PreActResidualBlock(channels=64)

        assert isinstance(block.bn1, torch.nn.GroupNorm)
        assert isinstance(block.conv1, torch.nn.Conv2d)
        assert isinstance(block.bn2, torch.nn.GroupNorm)
        assert isinstance(block.conv2, torch.nn.Conv2d)
        assert isinstance(block.relu, torch.nn.LeakyReLU)
        # Pre-activation: BN comes before conv
        assert hasattr(block, "bn1")
        assert hasattr(block, "conv1")

    def test_init_stride_creates_shortcut(self):
        """Test that stride > 1 creates a shortcut convolution."""
        block_stride1 = PreActResidualBlock(channels=64, stride=1)
        block_stride2 = PreActResidualBlock(channels=64, stride=2)

        # stride=1 should have identity shortcut
        assert len(block_stride1.shortcut) == 0

        # stride=2 should have conv shortcut
        assert len(block_stride2.shortcut) == 2  # Conv2d + GroupNorm

    def test_forward_pass_identity_shortcut(self):
        """Test forward pass with identity shortcut."""
        block = PreActResidualBlock(channels=64, stride=1)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_projection_shortcut(self):
        """Test forward pass with projection shortcut."""
        block = PreActResidualBlock(channels=64, stride=2)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        # Should downsample spatially
        assert output.shape == (2, 64, 8, 8)
        assert isinstance(output, torch.Tensor)

    def test_pre_activation_vs_post_activation(self):
        """Test that pre-activation differs from post-activation."""
        preact_block = PreActResidualBlock(channels=64)
        postact_block = ResidualBlock(channels=64)

        x = torch.randn(1, 64, 8, 8)

        preact_out = preact_block(x)
        postact_out = postact_block(x)

        # Both should preserve shape
        assert preact_out.shape == x.shape
        assert postact_out.shape == x.shape
        # But outputs should be different due to different activation placement
        assert not torch.equal(preact_out, postact_out)


class TestUnifiedResidualBlock:
    """Test UnifiedResidualBlock."""

    def test_init_default_parameters(self):
        """Test UnifiedResidualBlock initialization with defaults."""
        block = UnifiedResidualBlock(channels=64)

        assert block.skip_scale == 1.0
        assert block.use_shortcut is True
        assert block.pre_activation is False
        assert isinstance(block.act, torch.nn.LeakyReLU)  # default activation
        assert isinstance(block.conv1, torch.nn.Conv2d)
        assert isinstance(block.conv2, torch.nn.Conv2d)

    def test_init_custom_parameters(self):
        """Test UnifiedResidualBlock initialization with custom parameters."""
        block = UnifiedResidualBlock(
            channels=128,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=True,
            use_group_norm=False,
            activation="leaky",
            skip_scale=0.5,
            use_shortcut=True,
            pre_activation=True,
        )

        assert block.skip_scale == 0.5
        assert block.use_shortcut is True
        assert block.pre_activation is True
        assert isinstance(block.act, torch.nn.LeakyReLU)
        assert block.conv1.bias is not None  # bias enabled
        assert isinstance(block.bn1, torch.nn.Identity)  # group norm disabled

    def test_init_activation_options(self):
        """Test different activation options."""
        relu_block = UnifiedResidualBlock(channels=64, activation="relu")
        leaky_block = UnifiedResidualBlock(channels=64, activation="leaky")
        gelu_block = UnifiedResidualBlock(channels=64, activation="gelu")

        assert isinstance(relu_block.act, torch.nn.ReLU)
        assert isinstance(leaky_block.act, torch.nn.LeakyReLU)
        assert isinstance(gelu_block.act, torch.nn.GELU)

    def test_init_no_shortcut(self):
        """Test UnifiedResidualBlock without shortcut."""
        block = UnifiedResidualBlock(channels=64, use_shortcut=False)

        assert block.use_shortcut is False
        assert isinstance(block.shortcut, torch.nn.Identity)

    def test_forward_pass_standard_mode(self):
        """Test forward pass in standard (post-activation) mode."""
        block = UnifiedResidualBlock(channels=64, pre_activation=False)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_pre_activation_mode(self):
        """Test forward pass in pre-activation mode."""
        block = UnifiedResidualBlock(channels=64, pre_activation=True)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_with_stride(self):
        """Test forward pass with stride (spatial downsampling)."""
        block = UnifiedResidualBlock(channels=64, stride=2)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == (2, 64, 8, 8)
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_skip_scale(self):
        """Test forward pass with skip scaling."""
        block = UnifiedResidualBlock(channels=64, skip_scale=0.1)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_no_shortcut(self):
        """Test forward pass without shortcut connection."""
        block = UnifiedResidualBlock(channels=64, use_shortcut=False)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)

        assert output.shape == x.shape
        assert isinstance(output, torch.Tensor)


class TestResidualBlocksIntegration:
    """Integration tests for residual blocks."""

    def test_residual_blocks_output_shapes(self):
        """Test that all residual blocks preserve channels."""
        channels, batch_size = 64, 2

        # Test different residual blocks
        blocks = [
            ResidualBlock(channels),
            PreActResidualBlock(channels),
            UnifiedResidualBlock(channels),
            UnifiedResidualBlock(channels, pre_activation=True),
        ]

        for block in blocks:
            x = torch.randn(batch_size, channels, 16, 16)
            output = block(x)
            assert output.shape[0] == batch_size
            assert output.shape[1] == channels  # channels preserved

    def test_residual_blocks_gradient_flow(self):
        """Test that gradients flow through residual blocks."""
        blocks = [
            ResidualBlock(channels=32),
            PreActResidualBlock(channels=32),
            UnifiedResidualBlock(channels=32),
        ]

        for block in blocks:
            x = torch.randn(2, 32, 16, 16)
            x.requires_grad_(True)

            output = block(x)
            loss = output.sum()
            loss.backward()

            assert x.grad is not None
            assert x.grad.shape == x.shape

    def test_residual_vs_no_residual(self):
        """Test that residual connections improve gradient flow."""
        # This is more of a conceptual test - residual blocks should help with
        # gradient flow in deep networks, but we can't easily test this
        # directly
        residual_block = ResidualBlock(channels=64)
        plain_conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.GroupNorm(32, 64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.GroupNorm(32, 64),
            torch.nn.ReLU(),
        )

        x = torch.randn(1, 64, 8, 8)

        residual_out = residual_block(x)
        plain_out = plain_conv(x)

        # Both should produce valid outputs
        assert residual_out.shape == x.shape
        assert plain_out.shape == x.shape


class TestResidualBlocksEdgeCases:
    """Test edge cases and error conditions for residual blocks."""

    def test_residual_block_minimal_channels(self):
        """Test residual blocks with minimal channels."""
        blocks = [
            ResidualBlock(channels=8),
            PreActResidualBlock(channels=8),
            UnifiedResidualBlock(channels=8),
        ]

        for block in blocks:
            x = torch.randn(1, 8, 4, 4)
            output = block(x)
            assert output.shape == x.shape

    def test_unified_residual_different_activations(self):
        """Test UnifiedResidualBlock with different activation functions."""
        activations = ["relu", "leaky", "gelu"]

        for act in activations:
            block = UnifiedResidualBlock(channels=32, activation=act)
            x = torch.randn(1, 32, 8, 8)
            output = block(x)
            assert output.shape == x.shape

    def test_unified_residual_skip_scale_zero(self):
        """Test UnifiedResidualBlock with zero skip scale."""
        block = UnifiedResidualBlock(channels=64, skip_scale=0.0)
        x = torch.randn(2, 64, 16, 16)

        output = block(x)
        assert output.shape == x.shape

        # With skip_scale=0, the residual connection is effectively removed
        # The output should still be valid but different from identity
        assert isinstance(output, torch.Tensor)

    def test_residual_blocks_different_kernel_sizes(self):
        """Test residual blocks with different kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            block = UnifiedResidualBlock(channels=64, kernel_size=kernel_size)
            x = torch.randn(1, 64, 16, 16)
            output = block(x)
            assert output.shape == x.shape
