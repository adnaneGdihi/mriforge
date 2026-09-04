"""Unit tests for Soft-DTW alignment loss."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.soft_dtw_loss import SoftDTWLoss, _soft_dtw_distance


class TestSoftDTWDistance:
    """Test the raw Soft-DTW distance computation."""

    def test_identity(self):
        """DTW(x, x) should be approximately 0."""
        x = torch.randn(2, 16, 4)
        dist = _soft_dtw_distance(x, x, gamma=1.0)
        assert dist.shape == (2,)
        assert (dist < 1.0).all()  # Should be near 0

    def test_symmetry(self):
        """DTW(x, y) ≈ DTW(y, x) for matching lengths."""
        x = torch.randn(2, 16, 4)
        y = torch.randn(2, 16, 4)
        d_xy = _soft_dtw_distance(x, y, gamma=1.0)
        d_yx = _soft_dtw_distance(y, x, gamma=1.0)
        assert torch.allclose(d_xy, d_yx, atol=0.5)

    def test_output_shape(self):
        pred = torch.randn(3, 10, 8)
        tgt = torch.randn(3, 10, 8)
        dist = _soft_dtw_distance(pred, tgt, gamma=0.1)
        assert dist.shape == (3,)


class TestSoftDTWLoss:
    """Test the SoftDTWLoss module."""

    def test_4d_input(self):
        """Should handle [B, C, H, W] input."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=8)
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        tgt = torch.randn(2, 1, 8, 8)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()  # Scalar
        loss.backward()
        assert pred.grad is not None
        # Stronger than existence: a detached/zero grad would pass `is not None`
        # yet mean the differentiable path is inert. Pin real signal.
        assert torch.any(pred.grad != 0)

    def test_3d_input(self):
        """Should handle [B, L, D] input."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=8)
        pred = torch.randn(2, 16, 4, requires_grad=True)
        tgt = torch.randn(2, 16, 4)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()

    def test_short_sequence(self):
        """When sequence < window_size, should compute on full sequence."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=128)
        pred = torch.randn(2, 16, 4, requires_grad=True)
        tgt = torch.randn(2, 16, 4)
        loss = loss_fn(pred, tgt)
        assert loss.shape == ()

    def test_invalid_dims(self):
        """Should raise on a 2-D input (no channel/spatial structure).

        5-D volumetric input is now SUPPORTED (folded slice-wise) — see
        :meth:`test_5d_volumetric_input_folds_to_slices`.
        """
        loss_fn = SoftDTWLoss()
        with pytest.raises(ValueError, match="expects 3D, 4D, or 5D"):
            loss_fn(torch.randn(2, 8), torch.randn(2, 8))

    def test_5d_volumetric_input_folds_to_slices(self):
        """A 5-D ``[B, C, H, W, D]`` volume (TorchIO depth-last) is aligned
        slice-wise. Pre-2026-06 this raised and crashed the volumetric
        ``exp_hm_08_sdtw_mamba`` arm (hdsf_mamba_unet backbone, 5-D output).
        """
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        pred = torch.randn(2, 1, 32, 32, 4)
        target = pred + 0.1 * torch.randn_like(pred)
        out = loss_fn(pred, target)
        assert out.shape == ()
        assert torch.isfinite(out)

    def test_5d_depth1_matches_squeezed_4d(self):
        """A single-slice 5-D volume equals the 4-D Soft-DTW of its squeezed
        slice — the depth fold is a no-op when ``D == 1``."""
        loss_fn = SoftDTWLoss(gamma=0.1, window_size=64)
        pred4 = torch.randn(2, 1, 16, 16)
        target4 = torch.randn(2, 1, 16, 16)
        out5 = loss_fn(pred4.unsqueeze(-1), target4.unsqueeze(-1))
        out4 = loss_fn(pred4, target4)
        assert torch.allclose(out5, out4, atol=1e-5)

    def test_registration(self):
        """Soft-DTW loss should be in the loss registry."""
        from spectramr.models.losses.registry import list_available

        available = list_available()
        assert "soft_dtw" in available
