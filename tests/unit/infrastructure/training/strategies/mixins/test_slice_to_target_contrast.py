"""Regression tests for ``MetricsMixin._slice_to_target_contrast_single``.

The slicer interprets ``data.target_channels`` as the per-contrast channel
count of a multi-contrast k-space tensor. Older code applied the slice
unconditionally — including when ``target_channels=1`` (the schema-default
"1-channel RSS magnitude target" semantic). For single-contrast multi-coil
data that meant the slicer kept *only the imaginary part of the last coil*
and the visualization stack rendered a meaningless k-space slice — read by
users as the experiment_11 "doubled-brain" / "white-spot" artefact.

Guard:
    - target_channels < 4 => never slice (single-contrast data).
    - target_channels >= 4 AND tensor.shape[1] is a clean multiple of it =>
      slice to last ``target_channels`` channels (real multi-contrast layout).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from mriforge.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin


def _make_strategy(target_channels: int | None) -> MetricsMixin:
    """Construct a bare MetricsMixin instance with the minimum config to exercise the slicer."""
    s = MetricsMixin.__new__(MetricsMixin)
    s.config = SimpleNamespace(data=SimpleNamespace(target_channels=target_channels))
    return s


class TestSliceToTargetContrastSingle:
    def test_single_contrast_8ch_target1_returns_unchanged(self):
        """experiment_11 variants: 8 real-stacked k-space ch (4 coils × R/I), target_channels=1.
        Slicer must NOT take the last 1 channel — that's the imaginary half of
        the last coil, which renders as the doubled-brain / white-spot artefact.
        """
        s = _make_strategy(target_channels=1)
        x = torch.randn(2, 8, 32, 32)
        out = s._slice_to_target_contrast_single(x)
        assert out.shape == x.shape, "target_channels=1 must skip slicing for multi-coil data"
        assert torch.equal(out, x)

    def test_single_contrast_4ch_target2_returns_unchanged(self):
        """2-coil case (4 ch real-stacked) with target_channels=2 must also skip —
        slicing here would keep only the last R/I pair (one coil) and drop coil
        information needed for proper RSS magnitude.
        """
        s = _make_strategy(target_channels=2)
        x = torch.randn(2, 4, 32, 32)
        out = s._slice_to_target_contrast_single(x)
        assert out.shape == x.shape

    def test_target_channels_none_returns_unchanged(self):
        s = _make_strategy(target_channels=None)
        x = torch.randn(1, 8, 16, 16)
        out = s._slice_to_target_contrast_single(x)
        assert torch.equal(out, x)

    def test_multi_contrast_16ch_target8_keeps_last_half(self):
        """Cross-contrast layout: 16 ch = 8 ch (T1 prior) || 8 ch (target).
        Slicer must keep the LAST 8 channels (the target contrast).
        """
        s = _make_strategy(target_channels=8)
        x = torch.arange(2 * 16 * 4 * 4, dtype=torch.float32).reshape(2, 16, 4, 4)
        out = s._slice_to_target_contrast_single(x)
        assert out.shape == (2, 8, 4, 4)
        # Last 8 channels of the input correspond to the target contrast
        assert torch.equal(out, x[:, 8:])

    def test_multi_contrast_24ch_target8_keeps_last_eight(self):
        """3-contrast layout (24 ch): keep the last 8."""
        s = _make_strategy(target_channels=8)
        x = torch.randn(1, 24, 8, 8)
        out = s._slice_to_target_contrast_single(x)
        assert out.shape == (1, 8, 8, 8)
        assert torch.equal(out, x[:, -8:])

    def test_target_channels_larger_than_tensor_no_op(self):
        s = _make_strategy(target_channels=8)
        x = torch.randn(1, 4, 8, 8)  # tensor has fewer channels than target_ch
        out = s._slice_to_target_contrast_single(x)
        assert torch.equal(out, x)

    def test_uneven_division_no_op(self):
        s = _make_strategy(target_channels=8)
        x = torch.randn(1, 12, 4, 4)  # 12 % 8 != 0 → don't slice
        out = s._slice_to_target_contrast_single(x)
        assert torch.equal(out, x)

    def test_3d_tensor_no_op(self):
        s = _make_strategy(target_channels=8)
        x = torch.randn(8, 4, 4)  # not 4D
        out = s._slice_to_target_contrast_single(x)
        assert torch.equal(out, x)

    def test_doubled_brain_signature_no_longer_emitted(self):
        """End-to-end signature check: with target_channels=1 and 8-ch real-
        stacked k-space, the magnitude image obtained by pairing R/I and
        applying iFFT must NOT be centro-symmetric (which is the visual
        signature of running iFFT on a single real channel).
        """
        torch.manual_seed(0)
        s = _make_strategy(target_channels=1)
        # Build a synthetic complex k-space with a known asymmetric image
        H, W = 16, 16
        img = torch.zeros(1, 4, H, W)
        # Place 4-coil "anatomy" off-centre so we can detect symmetry violations
        img[:, :, 4:8, 4:8] = 1.0
        img_complex = torch.complex(img, torch.zeros_like(img))
        # FFT to k-space (centred)
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
        ksp_complex = fft2c(img_complex)
        # Real-stacked layout [R0, I0, R1, I1, R2, I2, R3, I3] (8 channels)
        ksp_real = torch.zeros(1, 8, H, W)
        ksp_real[:, 0::2] = ksp_complex.real
        ksp_real[:, 1::2] = ksp_complex.imag

        # Slicer is a no-op (target_ch=1 < 4)
        sliced = s._slice_to_target_contrast_single(ksp_real)
        assert sliced.shape == ksp_real.shape

        # Pair as complex and iFFT (the path the visualization actually takes)
        complex_paired = torch.complex(sliced[:, 0::2], sliced[:, 1::2])
        recon = ifft2c(complex_paired).abs()
        # RSS over the 4 coil dim
        rss = torch.sqrt((recon ** 2).sum(dim=1, keepdim=True))

        # Centro-symmetric magnitude would have rss[i, j] ≈ rss[H-1-i, W-1-j].
        # Our anatomy is in the upper-left quadrant; the lower-right quadrant
        # should be near-zero. Pre-fix, both quadrants would be bright.
        upper_left = rss[:, :, 4:8, 4:8].mean()
        lower_right = rss[:, :, H - 8:H - 4, W - 8:W - 4].mean()
        assert upper_left > 5 * lower_right, (
            f"doubled-brain signature detected: upper_left={upper_left.item():.3f}, "
            f"lower_right={lower_right.item():.3f}"
        )
