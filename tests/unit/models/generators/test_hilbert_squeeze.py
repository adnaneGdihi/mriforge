"""Unit tests for _squeeze_volume_to_2d / _unsqueeze_to_volume helpers.

These helpers enable Hilbert-Mamba generators (which use Conv2d stems) to
gracefully handle 5D volume inputs from TorchIO SubjectsLoader during
validation, where full 3D NIfTI volumes are returned instead of 2D patches.
"""

import torch

from spectramr.models.generators.hilbert_mamba_generators import (
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)


class TestSqueezeVolumeTo2D:
    """Tests for _squeeze_volume_to_2d."""

    def test_noop_4d(self) -> None:
        """4D input [B, C, H, W] passes through unchanged."""
        x = torch.randn(2, 1, 64, 64)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (2, 1, 64, 64)
        assert vol_shape is None
        assert squeezed is x  # Same object, no copy

    def test_squeeze_last_dim_1(self) -> None:
        """5D input [B, C, H, W, 1] squeezes last dim (TorchIO 2D patch)."""
        x = torch.randn(2, 1, 64, 64, 1)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (2, 1, 64, 64)
        assert vol_shape == (2, 1, 64, 64, 1)

    def test_middle_slice_5d(self) -> None:
        """5D input [B, C, D, H, W] takes middle slice along D."""
        x = torch.randn(2, 1, 256, 64, 64)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (2, 1, 64, 64)
        assert vol_shape == (2, 1, 256, 64, 64)
        # Verify it's the middle slice (D=256, mid=128)
        assert torch.equal(squeezed, x[:, :, 128])

    def test_middle_slice_odd_depth(self) -> None:
        """Middle slice with odd depth dimension."""
        x = torch.randn(1, 2, 7, 32, 32)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (1, 2, 32, 32)
        assert torch.equal(squeezed, x[:, :, 3])  # 7 // 2 = 3

    def test_3d_input_raises(self) -> None:
        """3D input is left unchanged (no squeeze needed)."""
        x = torch.randn(2, 64)  # 2D, not handled
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert vol_shape is None


class TestUnsqueezeToVolume:
    """Tests for _unsqueeze_to_volume."""

    def test_noop_none_shape(self) -> None:
        """None vol_shape returns output unchanged."""
        out = torch.randn(2, 1, 64, 64)
        result = _unsqueeze_to_volume(out, None)
        assert result is out

    def test_unsqueeze_last_dim_1(self) -> None:
        """Restores [B, C, H, W] → [B, C, H, W, 1]."""
        out = torch.randn(2, 1, 64, 64)
        original_shape = (2, 1, 64, 64, 1)
        result = _unsqueeze_to_volume(out, original_shape)
        assert result.shape == (2, 1, 64, 64, 1)
        assert torch.equal(result.squeeze(-1), out)

    def test_expand_middle_slice(self) -> None:
        """Restores middle-slice to full volume via expand."""
        out = torch.randn(2, 1, 64, 64)
        original_shape = (2, 1, 256, 64, 64)
        result = _unsqueeze_to_volume(out, original_shape)
        assert result.shape == (2, 1, 256, 64, 64)
        # All slices should be the same prediction (broadcast)
        assert torch.equal(result[:, :, 0], result[:, :, 128])


class TestSqueezeRoundtrip:
    """End-to-end squeeze → unsqueeze roundtrip tests."""

    def test_roundtrip_4d(self) -> None:
        """4D input roundtrips exactly."""
        x = torch.randn(2, 1, 64, 64)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        restored = _unsqueeze_to_volume(squeezed, vol_shape)
        assert torch.equal(x, restored)

    def test_roundtrip_5d_depth_1(self) -> None:
        """5D [B,C,H,W,1] roundtrips exactly."""
        x = torch.randn(2, 1, 64, 64, 1)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        restored = _unsqueeze_to_volume(squeezed, vol_shape)
        assert torch.equal(x, restored)

    def test_roundtrip_5d_depth_n(self) -> None:
        """5D [B,C,D,H,W] round-trips with consistent shape."""
        x = torch.randn(2, 1, 128, 32, 32)
        squeezed, vol_shape = _squeeze_volume_to_2d(x)
        assert squeezed.shape == (2, 1, 32, 32)
        restored = _unsqueeze_to_volume(squeezed, vol_shape)
        assert restored.shape == x.shape
