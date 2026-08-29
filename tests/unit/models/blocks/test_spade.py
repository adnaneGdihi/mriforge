"""Unit tests for SPADEBlock and SPADEEncoder.

Tests the spatially-adaptive feature modulation blocks used for
injecting marker priors into the decoder.
"""

import pytest
import torch

from mriforge.models.blocks.spade import SPADEBlock, SPADEEncoder


class TestSPADEBlock:
    """Tests for the core SPADE modulation block."""

    def test_output_shape_preserved(self):
        """Output shape should match feature map shape."""
        spade = SPADEBlock(feature_channels=64, marker_channels=1)
        h = torch.randn(2, 64, 32, 32)
        marker = torch.randn(2, 1, 32, 32)
        out = spade(h, marker)
        assert out.shape == h.shape

    def test_marker_spatial_resize(self):
        """Marker at different resolution should be resized automatically."""
        spade = SPADEBlock(feature_channels=64, marker_channels=1)
        h = torch.randn(1, 64, 16, 16)
        marker = torch.randn(1, 1, 64, 64)  # Higher-res marker
        out = spade(h, marker)
        assert out.shape == h.shape

    def test_different_markers_different_outputs(self):
        """Different marker inputs should produce different modulations."""
        spade = SPADEBlock(feature_channels=32, marker_channels=1)
        h = torch.randn(1, 32, 16, 16)
        m1 = torch.randn(1, 1, 16, 16)
        m2 = torch.randn(1, 1, 16, 16) * 5.0

        out1 = spade(h, m1)
        out2 = spade(h, m2)
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_gradient_flows_through_marker(self):
        """Gradients should flow through the marker path."""
        spade = SPADEBlock(feature_channels=32, marker_channels=2)
        h = torch.randn(1, 32, 8, 8, requires_grad=True)
        marker = torch.randn(1, 2, 8, 8, requires_grad=True)

        out = spade(h, marker)
        loss = out.sum()
        loss.backward()

        assert marker.grad is not None
        assert h.grad is not None
        assert marker.grad.abs().sum() > 0

    def test_gradient_flows_through_features(self):
        """Gradients should flow through the feature map path."""
        spade = SPADEBlock(feature_channels=16, marker_channels=1)
        h = torch.randn(1, 16, 8, 8, requires_grad=True)
        marker = torch.randn(1, 1, 8, 8)

        out = spade(h, marker)
        loss = out.sum()
        loss.backward()

        assert h.grad is not None

    def test_multi_channel_marker(self):
        """Multi-channel markers (e.g., structural priors) should work."""
        spade = SPADEBlock(feature_channels=64, marker_channels=4)
        h = torch.randn(1, 64, 16, 16)
        marker = torch.randn(1, 4, 16, 16)
        out = spade(h, marker)
        assert out.shape == h.shape

    def test_instance_norm_variant(self):
        """InstanceNorm variant should work without errors."""
        spade = SPADEBlock(
            feature_channels=32, marker_channels=1, norm_type="instance",
        )
        h = torch.randn(2, 32, 16, 16)
        marker = torch.randn(2, 1, 16, 16)
        out = spade(h, marker)
        assert out.shape == h.shape


class TestSPADEEncoder:
    """Tests for the multi-scale SPADE encoder."""

    def test_multi_level_modulation(self):
        """Each level should produce correct output shapes."""
        channels = [128, 64, 32]
        encoder = SPADEEncoder(marker_channels=1, feature_channels_per_level=channels)

        marker = torch.randn(1, 1, 64, 64)
        for level, ch in enumerate(channels):
            size = 8 * (2 ** level)  # 8, 16, 32
            h = torch.randn(1, ch, size, size)
            out = encoder(h, marker, level)
            assert out.shape == h.shape

    def test_invalid_level_raises(self):
        """Accessing non-existent level should raise IndexError."""
        encoder = SPADEEncoder(marker_channels=1, feature_channels_per_level=[64, 32])

        with pytest.raises(IndexError, match="has 2 levels"):
            encoder(torch.randn(1, 64, 8, 8), torch.randn(1, 1, 8, 8), level=5)

    def test_num_levels_property(self):
        """num_levels should match the number of channel specs."""
        encoder = SPADEEncoder(marker_channels=1, feature_channels_per_level=[128, 64, 32, 16])
        assert encoder.num_levels == 4
