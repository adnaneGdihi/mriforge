"""Unit tests for MortonLocalityStem and QuadScanFusion.

Tests verify:
- Roundtrip: reverse_topology(forward(x)) preserves information
- Locality: Morton ordering reduces average sequence distance between
  2D neighbours compared to raster scan
- 3D Morton codes compute correctly
- QuadScanFusion produces output of correct shape
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.morton_stem import (
    MortonLocalityStem,
    QuadScanFusion,
    _expand_bits_2d,
    compute_morton_2d,
    compute_morton_3d,
)
from tests.utils.optional_backends import requires_cuda_for_mamba


class TestExpandBits2D:
    """Test bit-interleaving helpers."""

    def test_zero(self):
        assert _expand_bits_2d(torch.tensor(0)).item() == 0

    def test_one(self):
        # 1 -> 0b01 -> expanded: 0b01
        assert _expand_bits_2d(torch.tensor(1)).item() == 1

    def test_two(self):
        # 2 -> 0b10 -> expanded: 0b0100
        assert _expand_bits_2d(torch.tensor(2)).item() == 4

    def test_three(self):
        # 3 -> 0b11 -> expanded: 0b0101
        assert _expand_bits_2d(torch.tensor(3)).item() == 5


class TestComputeMorton2D:
    """Test 2D Z-order permutation computation."""

    def test_output_shapes(self):
        fwd, inv = compute_morton_2d(4, 4)
        assert fwd.shape == (16,)
        assert inv.shape == (16,)

    def test_permutation_is_valid(self):
        """Forward permutation must be a valid permutation of [0, N)."""
        fwd, _ = compute_morton_2d(8, 8)
        assert set(fwd.tolist()) == set(range(64))

    def test_roundtrip(self):
        """inv(fwd) == identity."""
        fwd, inv = compute_morton_2d(8, 8)
        identity = torch.arange(64)
        roundtrip = identity[fwd][inv]
        assert torch.equal(roundtrip, identity)

    def test_non_square_grid(self):
        """Morton still works for non-square grids."""
        fwd, inv = compute_morton_2d(4, 8)
        assert fwd.shape == (32,)
        identity = torch.arange(32)
        assert torch.equal(identity[fwd][inv], identity)


class TestComputeMorton3D:
    """Test 3D Z-order permutation computation."""

    def test_output_shapes(self):
        fwd, inv = compute_morton_3d(2, 4, 4)
        assert fwd.shape == (32,)
        assert inv.shape == (32,)

    def test_roundtrip_3d(self):
        fwd, inv = compute_morton_3d(2, 4, 4)
        identity = torch.arange(32)
        assert torch.equal(identity[fwd][inv], identity)


class TestMortonLocalityStem:
    """Test the full locality stem module."""

    @pytest.fixture
    def stem(self):
        return MortonLocalityStem(
            in_channels=2, embed_dim=64, patch_size=4, image_size=(16, 16)
        )

    def test_forward_shape(self, stem):
        x = torch.randn(2, 2, 16, 16)
        out = stem(x)
        # grid = 16/4 = 4×4 = 16 tokens
        assert out.shape == (2, 16, 64)

    def test_reverse_topology_shape(self, stem):
        x = torch.randn(2, 2, 16, 16)
        seq = stem(x)
        img = stem.reverse_topology(seq)
        assert img.shape == (2, 64, 4, 4)

    def test_roundtrip_preserves_information(self, stem):
        """forward → reverse should preserve all patch embeddings (up to reordering)."""
        x = torch.randn(1, 2, 16, 16)
        seq = stem(x)
        restored = stem.reverse_topology(seq)
        # The restored patches should equal the forward-projected patches
        patches_direct = stem.proj(x)  # [1, 64, 4, 4]
        assert torch.allclose(restored, patches_direct, atol=1e-6)

    def test_morton_indices_are_buffers(self, stem):
        """Indices should not be parameters (no gradients)."""
        assert not stem.morton_fwd.requires_grad
        assert not stem.morton_inv.requires_grad

    def test_locality_improvement(self):
        """Morton ordering should reduce avg distance between vertical 2D neighbours.

        Raster scan separates vertical neighbours by W steps. Morton Z-order
        should significantly reduce this distance by grouping 2D-adjacent
        patches into the same quadrant.
        """
        H, W = 16, 16
        fwd, inv = compute_morton_2d(H, W)

        # inv[raster_idx] = position in the morton sequence
        raster_v_dists = []
        morton_v_dists = []

        for r in range(H - 1):
            for c in range(W):
                idx = r * W + c
                nidx = (r + 1) * W + c  # Vertical neighbour
                raster_v_dists.append(abs(idx - nidx))
                morton_v_dists.append(abs(inv[idx].item() - inv[nidx].item()))

        avg_raster = sum(raster_v_dists) / len(raster_v_dists)
        avg_morton = sum(morton_v_dists) / len(morton_v_dists)

        # Morton should improve vertical locality (expected ~29% improvement)
        assert avg_morton < avg_raster, (
            f"Morton avg vertical distance ({avg_morton:.1f}) should be less "
            f"than raster ({avg_raster:.1f})"
        )


class TestQuadScanFusion:
    """Test 4-directional cross-scan fusion."""

    @pytest.fixture
    def fusion(self):
        return QuadScanFusion(d_model=32)

    @requires_cuda_for_mamba
    def test_output_shape(self, fusion):
        seq = torch.randn(2, 16, 32)
        out = fusion(seq)
        assert out.shape == (2, 16, 32)

    @requires_cuda_for_mamba
    def test_with_explicit_directions(self, fusion):
        z_fwd = torch.randn(2, 16, 32)
        z_rev = torch.flip(z_fwd, dims=[1])
        z_tfwd = torch.randn(2, 16, 32)
        z_trev = torch.flip(z_tfwd, dims=[1])
        out = fusion(z_fwd, z_rev, z_tfwd, z_trev)
        assert out.shape == (2, 16, 32)

    def test_merge_weights_sum_to_one(self, fusion):
        """Softmax-normalized weights should sum to 1."""
        w = torch.softmax(fusion.merge_weight, dim=0)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)
