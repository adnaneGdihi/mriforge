"""Unit tests for Robust MRI PSNR metric.

Tests the three critical fixes:
1. Background Zero-Inflation prevention via ROI masking
2. Data Range accuracy using target.max()
3. Complex magnitude handling

References:
    - Issue: Signal Energy Collapse and PSNR inflation in Experiment 11/12
    - Fix: Robust MRI PSNR with tissue masking
"""

import pytest
import torch

from mriforge.core.metrics.evaluation_metrics import PSNR, RobustMRI_PSNR
from mriforge.core.metrics.outcome import MetricNotApplicableError


class TestRobustMRIPSNR:
    """Test suite for RobustMRI_PSNR metric."""

    def test_background_zero_inflation_fix(self):
        """Test that robust PSNR prevents background zero-inflation.

        Standard PSNR on MRI images with 70% background zeros will be
        artificially inflated. Robust PSNR should be lower (more accurate).
        """
        # Create synthetic MRI image: 256x256 with 70% background (zeros)
        B, C, H, W = 1, 1, 256, 256
        target = torch.zeros(B, C, H, W)

        # Add tissue region in center (30% of image)
        tissue_size = int(H * 0.55)  # 121x121 = ~22% of total
        start = (H - tissue_size) // 2
        end = start + tissue_size

        # Tissue has signal in range [0, 500] (realistic MRI magnitude)
        target[:, :, start:end, start:end] = (
            torch.rand(B, C, tissue_size, tissue_size) * 500
        )

        # Prediction: Perfect on background, 20% amplitude collapse on tissue (pred max = 100 vs target max = 500)
        pred = target.clone()
        pred[:, :, start:end, start:end] = target[:, :, start:end, start:end] * 0.2

        # Standard PSNR (computes over entire image).
        #
        # ``use_target_max=True``, not ``False`` + ``data_range=None``. This
        # target spans [0, 500], which honours neither the [0, 1] nor the
        # [-1, 1] contract, so auto-inference now RAISES rather than silently
        # assuming 1.0. The subject of this test is tissue-only vs whole-image
        # masking, so both metrics must use the SAME range for the comparison to
        # mean anything -- RobustMRI_PSNR uses target.max(), so this does too.
        standard_psnr = PSNR(device="cpu", use_target_max=True)
        psnr_standard = standard_psnr.compute_metric(pred, target)

        # Robust PSNR (only tissue region)
        robust_psnr = RobustMRI_PSNR(background_threshold=1e-5, device="cpu")
        psnr_robust = robust_psnr.compute_metric(pred, target)

        # Background zeros inflate standard PSNR or deflate it based on data_range.
        # Robust PSNR should be distinctly different because it only looks at tissue errors
        assert psnr_standard != psnr_robust, (
            f"Standard PSNR ({psnr_standard:.2f} dB) should differ from "
            f"Robust PSNR ({psnr_robust:.2f} dB)"
        )

        # With 80% amplitude error (pred=0.2*target), PSNR should be low
        # Theoretical: error = 0.8*target, MSE = 0.64*target^2
        # For random signal, RMS ~ max/sqrt(3), so PSNR ~ 20*log10(sqrt(3)/0.8) ~ 6.7 dB
        # Robust PSNR should catch this correctly (6-8 dB range)
        assert 0.0 <= psnr_robust < 20.0, (
            f"Robust PSNR ({psnr_robust:.2f} dB) outside expected range [0, 20] dB "
            f"for 80% amplitude error"
        )

    def test_data_range_hallucination_fix(self):
        """Test that robust PSNR uses correct data_range (target.max()).

        If PSNR assumes data_range=1.0 or 255 when actual signal max=100,
        the PSNR will be artificially inflated.
        """
        # Create target with max=100 (not 1.0 or 255)
        target = torch.rand(1, 1, 64, 64) * 100

        # Perfect prediction (should give PSNR ~100 dB)
        pred = target.clone()

        robust_psnr = RobustMRI_PSNR(device="cpu")
        psnr = robust_psnr.compute_metric(pred, target)

        # Perfect reconstruction should give very high PSNR (clamped at 100)
        assert (
            psnr >= 90.0
        ), f"Perfect reconstruction PSNR ({psnr:.2f} dB) should be >= 90 dB"

        # Now test with slight error
        pred_noisy = target + torch.randn_like(target) * 1.0  # SNR ~ 100
        psnr_noisy = robust_psnr.compute_metric(pred_noisy, target)

        # With noise std=1 and signal max=100, PSNR ~ 20*log10(100/1) = 40 dB
        assert (
            35.0 < psnr_noisy < 45.0
        ), f"Noisy PSNR ({psnr_noisy:.2f} dB) outside expected range"

    def test_complex_magnitude_handling(self):
        """Test that robust PSNR correctly handles complex MRI data."""
        # Complex-valued MRI image (k-space or phase-sensitive)
        B, C, H, W = 1, 1, 64, 64
        target_real = torch.rand(B, C, H, W) * 100
        target_imag = torch.rand(B, C, H, W) * 100
        target = torch.complex(target_real, target_imag)

        # Prediction with phase error but correct magnitude
        pred_real = target_real * 0.9
        pred_imag = target_imag * 0.9
        pred = torch.complex(pred_real, pred_imag)

        robust_psnr = RobustMRI_PSNR(device="cpu")
        psnr = robust_psnr.compute_metric(pred, target)

        # Should compute PSNR on magnitudes
        assert psnr > 0, "PSNR should be positive for complex inputs"
        assert not torch.isnan(psnr), "PSNR should not be NaN for complex inputs"
        assert not torch.isinf(psnr), "PSNR should not be Inf for complex inputs"

    def test_real_imag_channel_encoding(self):
        """Test [B, 2, H, W] real/imag channel encoding."""
        # MRI data with real/imag as separate channels
        B, H, W = 1, 64, 64
        target = torch.zeros(B, 2, H, W)
        target[:, 0] = torch.rand(B, H, W) * 100  # Real channel
        target[:, 1] = torch.rand(B, H, W) * 100  # Imag channel

        pred = target.clone()
        pred[:, 0] *= 0.8  # 20% error
        pred[:, 1] *= 0.8

        robust_psnr = RobustMRI_PSNR(device="cpu")
        psnr = robust_psnr.compute_metric(pred, target)

        assert psnr > 0, "PSNR should be positive for [B, 2, H, W] encoding"
        # 20% amplitude error -> ~14 dB PSNR: 20*log10(1/0.2) = 13.98
        # Account for magnitude computation: sqrt(0.8^2 + 0.8^2) = 1.13, so slightly higher PSNR
        assert (
            12.0 < psnr < 20.0
        ), f"PSNR ({psnr:.2f} dB) outside expected range for 20% error"

    def test_tissue_mask_adapts_to_signal_range(self):
        """Test that tissue mask adapts to signal range (5% of max)."""
        # Target with max=1000
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 16:48, 16:48] = torch.rand(1, 1, 32, 32) * 1000

        # Prediction with 50% amplitude collapse
        pred = target * 0.5

        robust_psnr = RobustMRI_PSNR(background_threshold=1e-5, device="cpu")
        psnr = robust_psnr.compute_metric(pred, target)

        # Adaptive threshold = 1000 * 0.05 = 50
        # Voxels > 50 are tissue, so most tissue voxels included
        # 50% amplitude error -> 20*log10(1/0.5) = 6.02 dB (theoretical)
        # In practice, with random signal, PSNR may be higher due to partial overlap
        assert (
            0.0 <= psnr < 20.0
        ), f"PSNR ({psnr:.2f} dB) for 50% error outside expected range"

    def test_no_tissue_fallback_to_standard_psnr(self):
        """Test fallback to standard PSNR when no tissue detected."""
        # All-zero image (no tissue)
        target = torch.zeros(1, 1, 64, 64)
        pred = torch.zeros(1, 1, 64, 64)

        robust_psnr = RobustMRI_PSNR(device="cpu")
        psnr = robust_psnr.compute_metric(pred, target)

        # Should return very high PSNR for identical images
        assert psnr >= 90.0, f"Zero images should give PSNR >= 90 dB, got {psnr:.2f}"

    def test_comparison_with_standard_psnr_on_uniform_image(self):
        """Test that robust PSNR matches standard PSNR on uniform images."""
        # Uniform image (no background) - should match standard PSNR
        target = torch.rand(1, 1, 64, 64) * 100
        pred = target + torch.randn_like(target) * 5.0

        standard_psnr = PSNR(device="cpu", use_target_max=True)
        robust_psnr = RobustMRI_PSNR(background_threshold=1e-5, device="cpu")

        psnr_std = standard_psnr.compute_metric(pred, target)
        psnr_robust = robust_psnr.compute_metric(pred, target)

        # Should be very close (within 1 dB) for uniform images
        assert abs(psnr_std - psnr_robust) < 1.5, (
            f"Standard PSNR ({psnr_std:.2f}) and Robust PSNR ({psnr_robust:.2f}) "
            f"should match for uniform images"
        )

    def test_energy_conservation_after_fft_ifft(self):
        """Test that FFT-IFFT roundtrip preserves energy (verifying Fix #1 in review)."""
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c

        # Create image with known energy
        image = torch.randn(1, 1, 64, 64, dtype=torch.complex64) * 100

        # Energy in image domain
        energy_image = torch.sum(torch.abs(image) ** 2).item()

        # FFT -> k-space
        kspace = fft2c(image)

        # Energy in k-space (with ortho normalization, should be equal)
        energy_kspace = torch.sum(torch.abs(kspace) ** 2).item()

        # IFFT -> back to image
        image_reconstructed = ifft2c(kspace)
        energy_reconstructed = torch.sum(torch.abs(image_reconstructed) ** 2).item()

        # Parseval's theorem: Energy should be conserved
        # With norm="ortho", energy is exactly preserved
        rel_error_kspace = abs(energy_image - energy_kspace) / energy_image
        rel_error_reconstructed = (
            abs(energy_image - energy_reconstructed) / energy_image
        )

        assert rel_error_kspace < 1e-5, (
            f"FFT energy conservation failed: "
            f"image={energy_image:.2f}, kspace={energy_kspace:.2f}, "
            f"rel_error={rel_error_kspace:.2e}"
        )

        assert rel_error_reconstructed < 1e-5, (
            f"IFFT roundtrip energy conservation failed: "
            f"image={energy_image:.2f}, reconstructed={energy_reconstructed:.2f}, "
            f"rel_error={rel_error_reconstructed:.2e}"
        )


class TestPSNRDataRangeFix:
    """Test standard PSNR class with use_target_max flag."""

    def test_use_target_max_flag(self):
        """Test that use_target_max=True prevents data_range hallucination."""
        # Target with max=100 (not normalized to [0, 1])
        target = torch.rand(1, 1, 64, 64) * 100
        pred = target + torch.randn_like(target) * 10.0

        # Without the flag, the range is UNRESOLVABLE and the metric refuses.
        #
        # This assertion used to be ``abs(val_maxmin - val_targetmax) > 30.0``,
        # justified in its own comment as "use_target_max=True uses 100, while
        # False defaults to 1.0!" -- i.e. it pinned the hallucination that
        # MetricNotApplicableError was added to stop. A 30 dB gap produced by an
        # invented data_range is not a property worth protecting.
        psnr_maxmin = PSNR(device="cpu", use_target_max=False)
        with pytest.raises(MetricNotApplicableError, match="data_range"):
            psnr_maxmin.compute_metric(pred, target)

        # With the flag, target.max() is used and the value is real.
        psnr_targetmax = PSNR(device="cpu", use_target_max=True)
        val_targetmax = psnr_targetmax.compute_metric(pred, target)

        # The invariant worth asserting: the flag AGREES with declaring the same
        # range by hand. "They differ" was only true because one side was wrong.
        val_explicit = PSNR(device="cpu").compute_metric(
            pred, target, data_range=float(target.max())
        )
        assert abs(val_targetmax - val_explicit) < 0.1, (
            f"use_target_max ({val_targetmax:.3f}) must agree with an explicit "
            f"data_range=target.max() ({val_explicit:.3f})"
        )

        # ...and it lands where an SNR~10 input should.
        assert 15.0 < val_targetmax < 25.0

    def test_explicit_data_range_override(self):
        """Test that explicit data_range parameter overrides auto-detection."""
        target = torch.rand(1, 1, 64, 64) * 100
        pred = target + torch.randn_like(target) * 10.0

        psnr_metric = PSNR(device="cpu")

        # An explicit range is honoured...
        psnr_explicit = psnr_metric.compute_metric(pred, target, data_range=100.0)
        assert 15.0 < psnr_explicit < 25.0

        # ...and WITHOUT one, the metric refuses rather than inventing 1.0.
        #
        # The old assertion was ``abs(psnr_explicit - psnr_auto) > 30.0``,
        # explained as "auto uses data_range=1.0 by default". That default is
        # the defect; a test asserting the two paths disagree by 30 dB was
        # pinning it in place.
        with pytest.raises(MetricNotApplicableError, match="data_range"):
            psnr_metric.compute_metric(pred, target)
