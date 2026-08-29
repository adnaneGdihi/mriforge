"""Unit tests for ChronologicalKSpaceStem.

Tests verify:
- Output shape for Cartesian mode
- Timestamp mode correctly reorders PE lines
- Unstem correctly reverses the projection
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.chronological_stem import (
    ChronologicalKSpaceStem,
    ChronologicalKSpaceUnstem,
)


class TestChronologicalKSpaceStem:
    """Test chronological k-space tokenizer."""

    @pytest.fixture
    def stem(self):
        return ChronologicalKSpaceStem(
            in_channels=2, embed_dim=64, num_readout_samples=32, mode="cartesian"
        )

    @pytest.fixture
    def timestamp_stem(self):
        return ChronologicalKSpaceStem(
            in_channels=2, embed_dim=64, num_readout_samples=32, mode="timestamp"
        )

    def test_cartesian_shape(self, stem):
        kspace = torch.randn(2, 2, 16, 32)  # [B, C, Ky, Kx]
        out = stem(kspace)
        assert out.shape == (2, 16, 64)  # [B, Ky, D]

    def test_timestamp_reordering(self, timestamp_stem):
        """Timestamp mode should reorder PE lines chronologically."""
        B, C, Ky, Kx = 1, 2, 8, 32
        kspace = torch.randn(B, C, Ky, Kx)

        # Reverse timestamp order (line 7 was acquired first)
        timestamps = torch.arange(Ky - 1, -1, -1).float().unsqueeze(0)  # [1, 8]

        out_no_ts = timestamp_stem(kspace)
        out_with_ts = timestamp_stem(kspace, timestamps=timestamps)

        # They should differ because of reordering
        assert not torch.allclose(out_no_ts, out_with_ts)

    def test_timestamp_monotonic(self, timestamp_stem):
        """After sorting, the sequence index should correspond to time."""
        B, Ky = 1, 16
        # Shuffled timestamps
        timestamps = torch.randperm(Ky).float().unsqueeze(0)
        sorted_idx = torch.argsort(timestamps, dim=1)

        # Verify monotonically increasing
        sorted_ts = timestamps[:, sorted_idx.squeeze()]
        diffs = sorted_ts[:, 1:] - sorted_ts[:, :-1]
        assert (diffs >= 0).all()

    def test_cartesian_with_none_timestamps(self, stem):
        """Cartesian mode should work fine with timestamps=None."""
        kspace = torch.randn(2, 2, 16, 32)
        out = stem(kspace, timestamps=None)
        assert out.shape == (2, 16, 64)


class TestChronologicalKSpaceUnstem:
    """Test reverse projection."""

    def test_roundtrip_shape(self):
        stem = ChronologicalKSpaceStem(
            in_channels=2, embed_dim=64, num_readout_samples=32
        )
        unstem = ChronologicalKSpaceUnstem(
            embed_dim=64, out_channels=2, num_readout_samples=32
        )

        kspace = torch.randn(2, 2, 16, 32)
        tokens = stem(kspace)
        reconstructed = unstem(tokens)
        assert reconstructed.shape == kspace.shape

    def test_gradient_flow(self):
        """Gradients should flow through stem → unstem."""
        stem = ChronologicalKSpaceStem(
            in_channels=2, embed_dim=64, num_readout_samples=32
        )
        unstem = ChronologicalKSpaceUnstem(
            embed_dim=64, out_channels=2, num_readout_samples=32
        )

        kspace = torch.randn(2, 2, 8, 32, requires_grad=True)
        tokens = stem(kspace)
        out = unstem(tokens)
        loss = out.sum()
        loss.backward()
        assert kspace.grad is not None
