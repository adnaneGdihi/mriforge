"""Unit tests for volume_inference.py (Task 5).

Tests the faithful per-slice inference port from the original
Baseline_scripts_inference.py::predict_volume.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mriforge.infrastructure.evaluation.mrixfields_baselines.generator_loader import LoadedBaseline
from mriforge.infrastructure.evaluation.mrixfields_baselines.volume_inference import (
    _center_crop_or_pad,
    apply_brain_mask,
    predict_volume,
)

# ---------------------------------------------------------------------------
# Helpers — build fake LoadedBaseline objects with no checkpoint needed
# ---------------------------------------------------------------------------


def _identity_baseline(crop_size=None):
    """LoadedBaseline whose forward is the identity (returns input unchanged)."""

    def _fwd(x: torch.Tensor) -> torch.Tensor:
        return x

    return LoadedBaseline(forward=_fwd, model_type="resnet", crop_size=crop_size, meta={})


def _tanh3_baseline(crop_size=None):
    """LoadedBaseline whose forward applies tanh(x * 3), i.e. non-identity."""

    def _fwd(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x * 3.0)

    return LoadedBaseline(forward=_fwd, model_type="resnet", crop_size=crop_size, meta={})


# ---------------------------------------------------------------------------
# _center_crop_or_pad tests
# ---------------------------------------------------------------------------


class TestCenterCropOrPad:
    def test_identity_same_shape(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = _center_crop_or_pad(a, (4, 4))
        np.testing.assert_array_equal(result, a)

    def test_pad_smaller_to_larger(self):
        a = np.ones((4, 4), dtype=np.float32)
        result = _center_crop_or_pad(a, (8, 8))
        assert result.shape == (8, 8)
        # original data should be in the center
        assert result.sum() == pytest.approx(16.0)
        # corners should be zero (padding)
        assert result[0, 0] == pytest.approx(0.0)

    def test_crop_larger_to_smaller(self):
        a = np.ones((8, 8), dtype=np.float32)
        result = _center_crop_or_pad(a, (4, 4))
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, np.ones((4, 4), dtype=np.float32))

    def test_round_trip_crop_then_uncrop(self):
        """Padding to larger then cropping back preserves non-zero center values."""
        a = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        padded = _center_crop_or_pad(a, (16, 16))
        assert padded.shape == (16, 16)
        restored = _center_crop_or_pad(padded, (8, 8))
        np.testing.assert_allclose(restored, a, atol=1e-6)


# ---------------------------------------------------------------------------
# apply_brain_mask tests
# ---------------------------------------------------------------------------


class TestApplyBrainMask:
    def test_zero_region_masked(self):
        """Background (source <= thresh) should be zeroed in prediction."""
        pred = np.ones((8, 8, 4), dtype=np.float32)
        source = np.ones((8, 8, 4), dtype=np.float32)
        # Zero out a region in source — this is the "background"
        source[0:3, 0:3, :] = 0.0
        result = apply_brain_mask(pred, source)
        # Background should be zeroed
        assert result[0:3, 0:3, :].sum() == pytest.approx(0.0)
        # Foreground should be unchanged
        assert result[4:, 4:, :].sum() == pytest.approx(pred[4:, 4:, :].sum())

    def test_all_foreground_unchanged(self):
        """When all source values are above thresh, prediction is unmodified."""
        pred = np.random.default_rng(1).random((6, 6, 3)).astype(np.float32)
        source = np.ones((6, 6, 3), dtype=np.float32) * 0.5
        result = apply_brain_mask(pred, source)
        np.testing.assert_array_equal(result, pred)

    def test_threshold_boundary(self):
        """Values exactly at thresh (1e-6) should be masked (not foreground)."""
        pred = np.ones((4, 4), dtype=np.float32)
        source = np.zeros((4, 4), dtype=np.float32)
        source[2, 2] = 1e-6  # at threshold — mask uses strict >
        result = apply_brain_mask(pred, source, thresh=1e-6)
        assert result[2, 2] == pytest.approx(0.0)  # 1e-6 > 1e-6 is False → masked

    def test_custom_threshold(self):
        """Custom thresh parameter is respected."""
        pred = np.ones((4, 4), dtype=np.float32)
        # Most values are 0.5 (above thresh=0.1); one value is below thresh.
        source = np.full((4, 4), 0.5, dtype=np.float32)
        source[0, 0] = 0.01  # below 0.1 thresh → should be masked
        result = apply_brain_mask(pred, source, thresh=0.1)
        assert result[0, 0] == pytest.approx(0.0)  # masked
        assert result[1, 1] == pytest.approx(1.0)  # above thresh → kept

    def test_dtype_preserved(self):
        pred = np.ones((4, 4), dtype=np.float32) * 0.7
        source = np.ones((4, 4), dtype=np.float32)
        result = apply_brain_mask(pred, source)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# predict_volume tests
# ---------------------------------------------------------------------------


class TestPredictVolume:
    """Tests for the faithful predict_volume port."""

    def test_identity_no_crop_roundtrip(self):
        """Identity forward + crop_size=None: [0,1] volume maps to itself."""
        rng = np.random.default_rng(42)
        vol = rng.random((8, 8, 4)).astype(np.float32)  # [0,1]
        loaded = _identity_baseline(crop_size=None)
        out = predict_volume(loaded, vol, slice_axis=2, device="cpu")
        assert out.shape == (8, 8, 4), f"shape mismatch: {out.shape}"
        # Identity forward: (v*2-1)*0.5+0.5 = v — perfect round-trip
        np.testing.assert_allclose(out, vol, atol=1e-5)

    def test_output_in_zero_one(self):
        """Output should always be clipped to [0, 1]."""
        vol = np.random.default_rng(7).random((6, 6, 3)).astype(np.float32)
        loaded = _identity_baseline(crop_size=None)
        out = predict_volume(loaded, vol, slice_axis=2, device="cpu")
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_shape_preserved_with_crop_size(self):
        """crop_size=(16,16) on [8,8,4] → output keeps original [8,8,4] shape."""
        vol = np.random.default_rng(3).random((8, 8, 4)).astype(np.float32)
        loaded = _identity_baseline(crop_size=(16, 16))
        out = predict_volume(loaded, vol, slice_axis=2, device="cpu")
        assert out.shape == (8, 8, 4), f"shape mismatch: {out.shape}"

    def test_crop_uncrop_roundtrip_identity(self):
        """Identity forward with crop/uncrop: original values are preserved."""
        rng = np.random.default_rng(9)
        vol = rng.random((8, 8, 4)).astype(np.float32)
        loaded = _identity_baseline(crop_size=(16, 16))
        out = predict_volume(loaded, vol, slice_axis=2, device="cpu")
        # Center region of padded slice is the original data — crops back correctly
        np.testing.assert_allclose(out, vol, atol=1e-5)

    def test_tanh3_forward_gives_different_output(self):
        """tanh(x*3) forward gives output != input — proves scaling is load-bearing (#16).

        If the x*2-1 / *0.5+0.5 round-trip were omitted, the forward would see
        values in [0,1] and the distinction would be lost.  This test guards that
        the scaling actually changes the input seen by the model.
        """
        rng = np.random.default_rng(11)
        vol = rng.random((8, 8, 4)).astype(np.float32)
        vol = np.clip(vol, 0.05, 0.95)  # avoid values where tanh ≈ identity
        loaded = _tanh3_baseline(crop_size=None)
        out = predict_volume(loaded, vol, slice_axis=2, device="cpu")
        assert out.shape == (8, 8, 4)
        # Output must differ from input — tanh(x*3) is NOT the identity
        assert not np.allclose(out, vol, atol=1e-3), (
            "tanh(x*3) output should differ from input; "
            "if it doesn't, the *2-1/*0.5+0.5 round-trip is broken"
        )

    def test_slice_axis_0(self):
        """Works with slice_axis=0 as well (iterates along first axis)."""
        vol = np.random.default_rng(5).random((4, 8, 8)).astype(np.float32)
        loaded = _identity_baseline(crop_size=None)
        out = predict_volume(loaded, vol, slice_axis=0, device="cpu")
        assert out.shape == (4, 8, 8)

    def test_slice_axis_1(self):
        """Works with slice_axis=1."""
        vol = np.random.default_rng(6).random((8, 4, 8)).astype(np.float32)
        loaded = _identity_baseline(crop_size=None)
        out = predict_volume(loaded, vol, slice_axis=1, device="cpu")
        assert out.shape == (8, 4, 8)
