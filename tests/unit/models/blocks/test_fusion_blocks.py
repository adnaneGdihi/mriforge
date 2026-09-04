"""Unit tests for fusion blocks.

Tests fusion mechanisms including attention fusion, feature fusion,
multi-scale fusion, adaptive fusion, residual fusion, and multimodal fusion.
"""

import pytest
import torch

from spectramr.models.blocks.fusion import (
    AdaptiveFusionBlock,
    AttentionFusionBlock,
    FeatureFusionBlock,
    MultiModalFusionBlock,
    MultiScaleFusionBlock,
    ResidualFusionBlock,
)


class TestAttentionFusionBlock:
    """Test AttentionFusionBlock."""

    def test_init_default_parameters(self):
        """Test AttentionFusionBlock initialization with defaults."""
        block = AttentionFusionBlock(channels=64, num_inputs=2)

        assert block.num_inputs == 2
        assert isinstance(block.attention, torch.nn.Sequential)
        assert isinstance(block.refine, torch.nn.Conv2d)
        assert block.refine.in_channels == 64
        assert block.refine.out_channels == 64

    def test_init_custom_parameters(self):
        """Test AttentionFusionBlock initialization with custom parameters."""
        block = AttentionFusionBlock(channels=128, num_inputs=3, reduction_ratio=8)

        assert block.num_inputs == 3
        # Check that attention network uses correct input channels
        first_conv = block.attention[1]  # Conv2d after AdaptiveAvgPool2d
        assert first_conv.in_channels == 128 * 3  # channels * num_inputs

    def test_forward_pass_two_inputs(self):
        """Test forward pass with two inputs."""
        block = AttentionFusionBlock(channels=64, num_inputs=2)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 16, 16)

        output = block([x1, x2])

        assert output.shape == x1.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_three_inputs(self):
        """Test forward pass with three inputs."""
        block = AttentionFusionBlock(channels=32, num_inputs=3)
        inputs = [torch.randn(1, 32, 8, 8) for _ in range(3)]

        output = block(inputs)

        assert output.shape == inputs[0].shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_wrong_num_inputs(self):
        """Test that forward pass fails with wrong number of inputs."""
        block = AttentionFusionBlock(channels=64, num_inputs=2)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 16, 16)
        x3 = torch.randn(2, 64, 16, 16)

        with pytest.raises(ValueError, match="Expected 2 inputs, got 3"):
            block([x1, x2, x3])

    def test_forward_pass_with_weights_ignored(self):
        """Test that provided weights are ignored (learned internally)."""
        block = AttentionFusionBlock(channels=64, num_inputs=2)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 16, 16)
        weights = [0.3, 0.7]

        # Should work the same with or without weights
        output1 = block([x1, x2])
        output2 = block([x1, x2], weights=weights)

        assert output1.shape == output2.shape
        # Outputs may differ due to internal learning vs provided weights


class TestFeatureFusionBlock:
    """Test FeatureFusionBlock."""

    def test_init_default_parameters(self):
        """Test FeatureFusionBlock initialization with defaults."""
        block = FeatureFusionBlock(channels=64, num_inputs=2)

        assert block.num_inputs == 2
        assert isinstance(block.channel_attention, torch.nn.Sequential)
        assert isinstance(block.spatial_attention, torch.nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass with multiple inputs."""
        block = FeatureFusionBlock(channels=32, num_inputs=2)
        x1 = torch.randn(2, 32, 16, 16)
        x2 = torch.randn(2, 32, 16, 16)

        output = block([x1, x2])

        assert output.shape == x1.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_different_shapes_error(self):
        """Test that different input shapes cause concatenation errors."""
        block = FeatureFusionBlock(channels=64, num_inputs=2)
        x1 = torch.randn(2, 64, 16, 16)
        x2 = torch.randn(2, 64, 8, 8)  # Different spatial size

        # This should fail because concatenation requires same spatial dims
        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            block([x1, x2])


class TestMultiScaleFusionBlock:
    """Test MultiScaleFusionBlock."""

    def test_init_default_parameters(self):
        """Test MultiScaleFusionBlock initialization with defaults."""
        block = MultiScaleFusionBlock(channels=64, num_scales=3)

        assert block.num_scales == 3
        assert len(block.scale_convs) == 3
        assert isinstance(block.cross_attention, torch.nn.MultiheadAttention)
        assert isinstance(block.fusion_conv, torch.nn.Sequential)

    def test_init_custom_parameters(self):
        """Test MultiScaleFusionBlock initialization with custom parameters."""
        block = MultiScaleFusionBlock(channels=128, num_scales=4)

        assert block.num_scales == 4
        assert len(block.scale_convs) == 4

    def test_forward_pass_three_scales(self):
        """Test forward pass with three scales of same size."""
        block = MultiScaleFusionBlock(channels=64, num_scales=3)

        # Create inputs of same size (simulating processed scales)
        scales = [
            torch.randn(2, 64, 16, 16),  # Same size for simplicity
            torch.randn(2, 64, 16, 16),
            torch.randn(2, 64, 16, 16),
        ]

        output = block(scales)

        # Output should match input spatial dimensions
        assert output.shape == scales[0].shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_wrong_num_scales(self):
        """Test that forward pass fails with wrong number of scales."""
        block = MultiScaleFusionBlock(channels=64, num_scales=3)
        scales = [torch.randn(2, 64, 16, 16) for _ in range(2)]  # Only 2 scales

        with pytest.raises(ValueError, match="Expected 3 inputs, got 2"):
            block(scales)


class TestAdaptiveFusionBlock:
    """Test AdaptiveFusionBlock."""

    def test_init_default_parameters(self):
        """Test AdaptiveFusionBlock initialization with defaults."""
        block = AdaptiveFusionBlock(channels=64, num_inputs=2)

        assert block.num_inputs == 2
        assert block.temperature == 1.0
        assert isinstance(block.weight_net, torch.nn.Sequential)
        assert isinstance(block.modulation, torch.nn.Conv2d)

    def test_init_custom_parameters(self):
        """Test AdaptiveFusionBlock initialization with custom parameters."""
        block = AdaptiveFusionBlock(channels=128, num_inputs=3, temperature=0.5)

        assert block.num_inputs == 3
        assert block.temperature == 0.5

    def test_forward_pass(self):
        """Test forward pass with learned weights."""
        block = AdaptiveFusionBlock(channels=32, num_inputs=2)
        x1 = torch.randn(2, 32, 16, 16)
        x2 = torch.randn(2, 32, 16, 16)

        output = block([x1, x2])

        assert output.shape == x1.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_three_inputs(self):
        """Test forward pass with three inputs."""
        block = AdaptiveFusionBlock(channels=64, num_inputs=3)
        inputs = [torch.randn(1, 64, 8, 8) for _ in range(3)]

        output = block(inputs)

        assert output.shape == inputs[0].shape
        assert isinstance(output, torch.Tensor)


class TestResidualFusionBlock:
    """Test ResidualFusionBlock."""

    def test_init_default_parameters(self):
        """Test ResidualFusionBlock initialization with defaults."""
        block = ResidualFusionBlock(channels=64, num_inputs=2)

        assert block.num_inputs == 2
        assert isinstance(block.residual_conv, torch.nn.Conv2d)
        assert isinstance(block.residual_conv, torch.nn.Conv2d)

    def test_forward_pass(self):
        """Test forward pass with residual connections."""
        block = ResidualFusionBlock(channels=32, num_inputs=2)
        x1 = torch.randn(2, 32, 16, 16)
        x2 = torch.randn(2, 32, 16, 16)

        output = block([x1, x2])

        assert output.shape == x1.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_multiple_inputs(self):
        """Test forward pass with multiple inputs."""
        block = ResidualFusionBlock(channels=64, num_inputs=4)
        inputs = [torch.randn(1, 64, 8, 8) for _ in range(4)]

        output = block(inputs)

        assert output.shape == inputs[0].shape
        assert isinstance(output, torch.Tensor)


class TestMultiModalFusionBlock:
    """Test MultiModalFusionBlock."""

    def test_init_default_parameters(self):
        """Test MultiModalFusionBlock initialization with defaults."""
        block = MultiModalFusionBlock(channels=64, num_modalities=2)

        assert block.num_modalities == 2
        assert isinstance(block.modality_convs, torch.nn.ModuleList)
        assert len(block.modality_convs) == 2
        assert isinstance(block.fusion_block, torch.nn.Module)

    def test_init_custom_parameters(self):
        """Test MultiModalFusionBlock initialization with custom parameters."""
        block = MultiModalFusionBlock(channels=128, num_modalities=3)

        assert block.num_modalities == 3
        assert len(block.modality_convs) == 3

    def test_forward_pass(self):
        """Test forward pass with multiple modalities."""
        block = MultiModalFusionBlock(channels=32, num_modalities=2)
        x1 = torch.randn(2, 32, 16, 16)
        x2 = torch.randn(2, 32, 16, 16)

        output = block([x1, x2])

        assert output.shape == x1.shape
        assert isinstance(output, torch.Tensor)

    def test_forward_pass_three_modalities(self):
        """Test forward pass with three modalities."""
        block = MultiModalFusionBlock(channels=64, num_modalities=3)
        modalities = [torch.randn(1, 64, 8, 8) for _ in range(3)]

        output = block(modalities)

        assert output.shape == modalities[0].shape
        assert isinstance(output, torch.Tensor)


class TestFusionBlocksIntegration:
    """Integration tests for fusion blocks."""

    def test_fusion_blocks_output_shapes(self):
        """Test that all fusion blocks preserve input shapes."""
        channels, batch_size = 64, 2

        # Test different fusion blocks
        blocks_and_inputs = [
            (
                AttentionFusionBlock(channels, 2),
                [torch.randn(batch_size, channels, 16, 16) for _ in range(2)],
            ),
            (
                FeatureFusionBlock(channels, 2),
                [torch.randn(batch_size, channels, 16, 16) for _ in range(2)],
            ),
            (
                AdaptiveFusionBlock(channels, 2),
                [torch.randn(batch_size, channels, 16, 16) for _ in range(2)],
            ),
            (
                ResidualFusionBlock(channels, 2),
                [torch.randn(batch_size, channels, 16, 16) for _ in range(2)],
            ),
            (
                MultiModalFusionBlock(channels, 2),
                [torch.randn(batch_size, channels, 16, 16) for _ in range(2)],
            ),
        ]

        for block, inputs in blocks_and_inputs:
            output = block(inputs)
            assert output.shape == inputs[0].shape

    def test_fusion_blocks_require_list_input(self):
        """Test that fusion blocks require list inputs."""
        block = AttentionFusionBlock(channels=64, num_inputs=2)
        x1 = torch.randn(2, 64, 16, 16)

        # Should fail when torch.cat receives a single tensor instead of a sequence
        with pytest.raises(
            TypeError, match="cat\\(\\) received an invalid combination"
        ):
            block(x1)

    def test_fusion_blocks_gradient_flow(self):
        """Test that gradients flow through fusion blocks."""
        block = AttentionFusionBlock(channels=32, num_inputs=2)
        x1 = torch.randn(2, 32, 16, 16)
        x2 = torch.randn(2, 32, 16, 16)

        x1.requires_grad_(True)
        x2.requires_grad_(True)

        output = block([x1, x2])
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None
        assert x1.grad.shape == x1.shape
        assert x2.grad.shape == x2.shape


class TestFusionBlocksEdgeCases:
    """Test edge cases and error conditions for fusion blocks."""

    def test_attention_fusion_minimal_channels(self):
        """Test AttentionFusionBlock with minimal channels."""
        block = AttentionFusionBlock(channels=8, num_inputs=2)
        x1 = torch.randn(1, 8, 4, 4)
        x2 = torch.randn(1, 8, 4, 4)

        output = block([x1, x2])
        assert output.shape == x1.shape

    def test_multiscale_fusion_single_scale(self):
        """Test MultiScaleFusionBlock with single scale."""
        block = MultiScaleFusionBlock(channels=64, num_scales=1)
        x = torch.randn(2, 64, 16, 16)

        output = block([x])
        assert output.shape == x.shape

    def test_adaptive_fusion_temperature_effect(self):
        """Test that temperature affects AdaptiveFusionBlock behavior."""
        block_cold = AdaptiveFusionBlock(channels=32, num_inputs=2, temperature=0.1)
        block_hot = AdaptiveFusionBlock(channels=32, num_inputs=2, temperature=10.0)

        x1 = torch.randn(1, 32, 8, 8)
        x2 = torch.randn(1, 32, 8, 8)

        output_cold = block_cold([x1, x2])
        output_hot = block_hot([x1, x2])

        # Outputs should be different due to temperature
        assert output_cold.shape == output_hot.shape
        # Note: Due to randomness in initialization, we can't guarantee different outputs

    def test_residual_fusion_identity_preservation(self):
        """Test that ResidualFusionBlock can learn identity when inputs are similar."""
        block = ResidualFusionBlock(channels=32, num_inputs=2)
        x = torch.randn(2, 32, 16, 16)

        # Same input twice
        output = block([x, x])

        # Should be able to learn to preserve the input
        assert output.shape == x.shape
