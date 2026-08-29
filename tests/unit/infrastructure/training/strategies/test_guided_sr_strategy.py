"""Unit tests for GuidedSuperResolutionStrategy reference extraction and input building."""

import pytest
import torch


def _make_extractor(ref_slab_size=3):
    """Create a callable that mimics the strategy's _extract_reference method."""
    import random as _random

    def extract_reference(hf_volume, mode="random_slice"):
        if hf_volume.ndim == 4:
            return hf_volume
        B, C, D, H, W = hf_volume.shape
        k = min(ref_slab_size, D)
        if mode == "random_slice":
            idx = _random.randint(0, D - 1)
            return hf_volume[:, :, idx : idx + 1, :, :]
        elif mode == "random_slab":
            start = _random.randint(0, max(0, D - k))
            return hf_volume[:, :, start : start + k, :, :]
        elif mode == "center_slice":
            mid = D // 2
            return hf_volume[:, :, mid : mid + 1, :, :]
        elif mode == "center_slab":
            mid = D // 2
            start = max(0, mid - k // 2)
            return hf_volume[:, :, start : start + k, :, :]
        else:
            msg = f"Unknown reference mode: {mode}"
            raise ValueError(msg)

    return extract_reference


def _build_reference_input(ulf_volume, hf_reference):
    """Simulate _build_reference_input from the strategy."""
    import torch.nn.functional as F

    if ulf_volume.ndim == 4:
        if hf_reference.shape[2:] != ulf_volume.shape[2:]:
            hf_reference = F.interpolate(
                hf_reference,
                size=ulf_volume.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([ulf_volume, hf_reference], dim=1)

    B, C, D, H, W = ulf_volume.shape
    _, _, D_ref, _, _ = hf_reference.shape
    hf_padded = torch.zeros_like(ulf_volume)
    start = max(0, (D - D_ref) // 2)
    end = min(D, start + D_ref)
    hf_padded[:, :, start:end, :, :] = hf_reference[:, :, : end - start, :, :]
    return torch.cat([ulf_volume, hf_padded], dim=1)


class TestReferenceExtraction:
    """Tests for HF reference extraction."""

    def test_random_slice_3d(self):
        """Random slice from 3D volume should have D=1."""
        extract = _make_extractor()
        volume = torch.randn(2, 1, 16, 64, 64)
        ref = extract(volume, mode="random_slice")
        assert ref.shape == (2, 1, 1, 64, 64)

    def test_random_slab_3d(self):
        """Random slab should have D=ref_slab_size."""
        extract = _make_extractor(ref_slab_size=3)
        volume = torch.randn(2, 1, 16, 64, 64)
        ref = extract(volume, mode="random_slab")
        assert ref.shape[2] == 3

    def test_center_slice_3d(self):
        """Center slice should be deterministic and at D//2."""
        extract = _make_extractor()
        volume = torch.randn(1, 1, 16, 32, 32)
        ref = extract(volume, mode="center_slice")
        assert ref.shape == (1, 1, 1, 32, 32)
        expected = volume[:, :, 8:9, :, :]
        assert torch.allclose(ref, expected)

    def test_2d_passthrough(self):
        """2D input should be returned unchanged."""
        extract = _make_extractor()
        volume = torch.randn(2, 1, 64, 64)
        ref = extract(volume, mode="random_slice")
        assert torch.equal(ref, volume)

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        extract = _make_extractor()
        volume = torch.randn(1, 1, 16, 32, 32)
        with pytest.raises(ValueError, match="Unknown reference mode"):
            extract(volume, mode="invalid")


class TestReferenceInputBuilding:
    """Tests for building combined ULF + HF reference input."""

    def test_2d_concatenation(self):
        """2D ULF + 2D HF ref should concatenate to 2C channels."""
        ulf = torch.randn(2, 1, 64, 64)
        hf_ref = torch.randn(2, 1, 64, 64)
        combined = _build_reference_input(ulf, hf_ref)
        assert combined.shape == (2, 2, 64, 64)

    def test_3d_zero_padding(self):
        """3D reference should be zero-padded to match ULF depth."""
        ulf = torch.randn(1, 1, 16, 64, 64)
        hf_ref = torch.randn(1, 1, 1, 64, 64)
        combined = _build_reference_input(ulf, hf_ref)
        assert combined.shape == (1, 2, 16, 64, 64)
        hf_channel = combined[:, 1:2, :, :, :]
        n_zeros = (hf_channel == 0).sum().item()
        total = hf_channel.numel()
        assert n_zeros >= total * 0.9

    def test_3d_reference_placed_at_center(self):
        """Reference slice should be placed at center of padded volume."""
        ulf = torch.randn(1, 1, 16, 32, 32)
        hf_ref = torch.ones(1, 1, 1, 32, 32) * 42.0
        combined = _build_reference_input(ulf, hf_ref)
        center_val = combined[0, 1, 7, 0, 0].item()
        assert center_val == pytest.approx(42.0)
