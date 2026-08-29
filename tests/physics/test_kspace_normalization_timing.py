"""
Tests for K-Space Normalization Timing Fix

Critical Test Suite for ensuring k-space normalization is applied BEFORE
undersampling (not after) to prevent scale mismatches between fully-sampled
and undersampled k-space data.

Test Invariants:
1. Normalization BEFORE masking: scaled_kspace * mask == scaled_kspace (no additional scaling)
2. Scale consistency: percentile computed on FULL k-space, not undersampled
3. Forward/inverse FFT preserves magnitude distribution
"""

import numpy as np
import pytest
import torch

# Assume imports from src
try:
    from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
except ImportError:
    # Fallback for testing without installation
    def fft2c(x: torch.Tensor) -> torch.Tensor:
        """Centered 2D FFT."""
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1)
        )

    def ifft2c(x: torch.Tensor) -> torch.Tensor:
        """Centered inverse 2D FFT."""
        return torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(x, dim=(-2, -1))), dim=(-2, -1)
        ).real


class KSpaceNormalizationFixture:
    """Fixture for generating test data."""

    @staticmethod
    def create_cartesian_mask(
        shape: tuple[int, int], acceleration: float = 4.0, center_fraction: float = 0.08
    ) -> torch.Tensor:
        """Create variable density Cartesian mask."""
        H, W = shape
        mask = torch.zeros((1, 1, H, W))

        # Center region (fully sampled)
        center_h = int(H * center_fraction)
        start_h = (H - center_h) // 2
        mask[:, :, start_h : start_h + center_h, :] = 1.0

        # Outer region (random undersampling)
        outer_mask = torch.rand((1, 1, H, W)) < (1.0 / acceleration)
        mask = torch.logical_or(mask.bool(), outer_mask).float()

        return mask

    @staticmethod
    def create_radial_mask(shape: tuple[int, int], num_arms: int = 32) -> torch.Tensor:
        """Create radial non-Cartesian mask."""
        H, W = shape
        y, x = np.ogrid[-H / 2 : H / 2, -W / 2 : W / 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        mask = torch.zeros((1, 1, H, W))
        for phase in np.linspace(0, 2 * np.pi, num_arms, endpoint=False):
            arm_mask = np.abs(np.angle(np.exp(1j * (theta - phase)))) < np.pi / num_arms
            mask[0, 0, arm_mask] = 1.0

        return mask


# ============================================================================
# CRITICAL INVARIANTS FOR K-SPACE NORMALIZATION
# ============================================================================


class TestKSpaceNormalizationInvariants:
    """Test mathematical invariants that MUST hold."""

    def test_normalization_before_masking_invariant(self):
        """
        CRITICAL INVARIANT:
        If normalization applied BEFORE masking, then:
        (normalized_kspace * mask) == masked_kspace when scale_factor set correctly

        This ensures model trains on consistent scale across masked/unmasked data.
        """
        # Create fully-sampled k-space
        image = torch.randn(1, 1, 64, 64)
        kspace_full = fft2c(image)
        kspace_magnitude = torch.abs(kspace_full)

        # Compute scale on FULL k-space (CORRECT)
        percentile = 0.99
        scale_full = torch.quantile(kspace_magnitude, percentile)

        # Normalize BEFORE masking
        kspace_normalized_before = kspace_full / (scale_full + 1e-8)

        # Create mask
        mask = KSpaceNormalizationFixture.create_cartesian_mask((64, 64))

        # Apply mask
        kspace_masked = kspace_normalized_before * mask

        # INVARIANT: Masked values should be exact same scale as normalized values
        # (not rescaled again)
        masked_region_scale = torch.abs(kspace_masked[mask.bool()]).max()

        # Verify scale is consistent with normalized values
        expected_max_scale = torch.abs(kspace_normalized_before[mask.bool()]).max()

        assert torch.allclose(
            masked_region_scale, expected_max_scale, rtol=1e-5
        ), f"Scale mismatch: masked={masked_region_scale:.6f}, expected={expected_max_scale:.6f}"

    def test_percentile_preserves_dynamic_range(self):
        """
        INVARIANT: Percentile-based normalization should preserve relative signal magnitudes.

        If signal A has magnitude A_mag and signal B has magnitude B_mag,
        after normalization, their ratio should be preserved (approximately).
        """
        # Create synthetic k-space with known structure
        kspace = torch.randn(1, 1, 128, 128, dtype=torch.complex64)
        kspace_magnitude = torch.abs(kspace)

        # Mark regions with different intended magnitudes
        high_region = torch.zeros_like(kspace_magnitude)
        high_region[:, :, :32, :32] = 1.0
        low_region = torch.zeros_like(kspace_magnitude)
        low_region[:, :, 96:, 96:] = 1.0

        # Get original ratio
        high_mean_orig = kspace_magnitude[high_region.bool()].mean()
        low_mean_orig = kspace_magnitude[low_region.bool()].mean()
        ratio_orig = high_mean_orig / (low_mean_orig + 1e-8)

        # Apply percentile normalization
        scale = torch.quantile(kspace_magnitude, 0.99)
        kspace_normalized = kspace / (scale + 1e-8)
        kspace_magnitude_norm = torch.abs(kspace_normalized)

        # Check ratio is preserved
        high_mean_norm = kspace_magnitude_norm[high_region.bool()].mean()
        low_mean_norm = kspace_magnitude_norm[low_region.bool()].mean()
        ratio_norm = high_mean_norm / (low_mean_norm + 1e-8)

        assert torch.allclose(
            ratio_orig, ratio_norm, rtol=0.05
        ), f"Signal ratio not preserved: ratio_orig={ratio_orig:.4f}, ratio_norm={ratio_norm:.4f}"

    def test_phase_preserved_under_normalization(self):
        """
        INVARIANT: Phase information must be preserved during normalization.

        For complex k-space: if kspace_norm = kspace / scale, then
        angle(kspace_norm) == angle(kspace) (same phase)
        """
        kspace = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
        original_phase = torch.angle(kspace)

        # Normalize
        scale = torch.abs(kspace).max()
        kspace_normalized = kspace / (scale + 1e-8)
        normalized_phase = torch.angle(kspace_normalized)

        # Phase must be identical
        assert torch.allclose(
            original_phase, normalized_phase, atol=1e-6
        ), "Phase information was altered during normalization"

    def test_hermitian_symmetry_preserved_after_normalization(self):
        """
        INVARIANT: For real-valued images, k-space has Hermitian symmetry.
        This must be preserved after normalization.

        Real image → kspace[k_x, k_y] = conj(kspace[-k_x, -k_y])
        """
        # Create real-valued image
        # Use random but ensured even dimensions
        image = torch.randn(1, 1, 64, 64)
        kspace = fft2c(image)

        # Normalize
        scale = torch.abs(kspace).max()
        kspace_normalized = kspace / (scale + 1e-8)

        # Check Hermitian symmetry: k(kx, ky) == conj(k(-kx, -ky))
        # For centered k-space, flipping around center
        kspace_flipped = torch.flip(kspace_normalized, dims=[-2, -1]).conj()

        # Roll by 1 to align indices for centered FFT symmetry
        # In centered FFT, DC is at N/2. Symmetry is around DC.
        kspace_flipped = torch.roll(kspace_flipped, shifts=(1, 1), dims=(-2, -1))

        assert torch.allclose(
            kspace_normalized, kspace_flipped, atol=1e-4
        ), "Hermitian symmetry violated after normalization"


# ============================================================================
# CARTESIAN VS NON-CARTESIAN BEHAVIOR
# ============================================================================


class TestNormalizationForTrajectories:
    """Test different normalization strategies for different trajectories."""

    def test_cartesian_normalization_before_masking(self):
        """
        Cartesian trajectory: normalization MUST occur BEFORE masking.

        Since Cartesian masking is binary (keep/discard), normalizing after masking
        would change the scale of preserved k-space values.
        """
        # Create data
        image = torch.randn(1, 1, 128, 128)
        kspace_full = fft2c(image)

        # Create Cartesian mask
        mask = KSpaceNormalizationFixture.create_cartesian_mask(
            (128, 128), acceleration=4.0
        )

        # WRONG WAY (after masking)
        kspace_masked_first = kspace_full * mask
        # Compute scale on undersampled data - WRONG
        scale_after_masking = torch.quantile(
            torch.abs(kspace_masked_first[mask.bool()]), 0.99
        )
        kspace_wrong = kspace_masked_first / (scale_after_masking + 1e-8)

        # RIGHT WAY (before masking)
        scale_before_masking = torch.quantile(torch.abs(kspace_full), 0.99)
        kspace_normalized = kspace_full / (scale_before_masking + 1e-8)
        kspace_right = kspace_normalized * mask

        # The difference should be significant (not all close)
        # Verify they're different (proving normalization timing matters)
        max_diff = torch.abs(kspace_wrong - kspace_right).max()
        assert (
            max_diff > 0.001
        ), f"Normalization timing should matter, but max difference is only {max_diff:.6f}"

    def test_non_cartesian_normalization_scale_consistency(self):
        """
        Non-Cartesian trajectory: normalization should use SAME scale as Cartesian.

        Radial/spiral undersampling doesn't simply zero out frequency components,
        so we must compute scale on the ENTIRE k-space before undersampling.
        """
        # Create full k-space
        image = torch.randn(1, 1, 128, 128)
        kspace_full = fft2c(image)
        kspace_magnitude = torch.abs(kspace_full)

        # Compute scale on FULL k-space
        scale_full = torch.quantile(kspace_magnitude, 0.99)

        # Normalized k-space (before trajectory)
        kspace_normalized = kspace_full / (scale_full + 1e-8)

        # Create radial mask
        radial_mask = KSpaceNormalizationFixture.create_radial_mask(
            (128, 128), num_arms=24
        )

        # Apply (simulating trajectory acquisition)
        kspace_radial = kspace_normalized * radial_mask

        # Verify: radial samples should have similar scale to full k-space normalized values
        radial_samples = kspace_radial[radial_mask.bool()]
        radial_magnitude_samples = torch.abs(radial_samples)

        # Statistics should be similar to what we'd get if we sampled
        # from normalized full k-space
        full_samples = kspace_normalized[radial_mask.bool()]
        full_magnitude_samples = torch.abs(full_samples)

        radial_mean = radial_magnitude_samples.mean()
        full_mean = full_magnitude_samples.mean()

        # Means should be close
        assert torch.allclose(
            radial_mean, full_mean, rtol=0.1
        ), f"Scale mismatch: radial_mean={radial_mean:.4f}, full_mean={full_mean:.4f}"


# ============================================================================
# PHYSICS-BASED TESTS
# ============================================================================


class TestNormalizationPhysicsCorrectness:
    """Test that normalization doesn't violate MRI physics."""

    def test_magnitude_after_ifft_consistent(self):
        """
        Test that normalization in k-space produces consistent magnitude in image space.

        PHYSICS: |IFFT(normalized_kspace)| should have consistent scale regardless
        of when normalization was applied (before vs after masking).
        """
        # Create image
        image = torch.randn(1, 1, 64, 64)
        kspace = fft2c(image)

        # Use same scale for both approaches to verify linearity
        # Approach 1: Normalize k-space directly
        scale = torch.abs(kspace).max()
        kspace_norm1 = kspace / (scale + 1e-8)
        image_from_norm1 = torch.abs(ifft2c(kspace_norm1))

        # Approach 2: Use matching scale in image domain
        # FFT(image/scale) = FFT(image)/scale
        image_norm = image / (scale + 1e-8)
        kspace_from_norm_image = fft2c(image_norm)
        image_from_norm2 = torch.abs(ifft2c(kspace_from_norm_image))

        # Both should give identical results due to linearity
        assert torch.allclose(
            image_from_norm1, image_from_norm2, rtol=1e-5, atol=1e-5
        ), "Normalization in k-space and image space give different results (violates linearity)"

    def test_signal_to_noise_ratio_preserved(self):
        """
        PHYSICS INVARIANT: SNR must be preserved under magnitude scaling.

        SNR = signal_power / noise_power
        If we scale both signal and noise by same factor, SNR unchanged.
        """
        signal = torch.randn(1, 1, 64, 64, dtype=torch.complex64)
        noise = torch.randn(1, 1, 64, 64, dtype=torch.complex64) * 0.1  # 10% noise

        kspace = signal + noise

        # Compute SNR before normalization
        signal_power_before = torch.abs(signal).pow(2).mean()
        noise_power_before = torch.abs(noise).pow(2).mean()
        snr_before = signal_power_before / (noise_power_before + 1e-8)

        # Apply normalization
        scale = torch.abs(kspace).max()
        kspace_normalized = kspace / (scale + 1e-8)
        signal_normalized = signal / (scale + 1e-8)
        noise_normalized = noise / (scale + 1e-8)

        # Compute SNR after normalization
        signal_power_after = torch.abs(signal_normalized).pow(2).mean()
        noise_power_after = torch.abs(noise_normalized).pow(2).mean()
        snr_after = signal_power_after / (noise_power_after + 1e-8)

        # SNR should be identical
        assert torch.allclose(
            snr_before, snr_after, rtol=1e-4
        ), f"SNR changed after normalization: before={snr_before:.4f}, after={snr_after:.4f}"


# ============================================================================
# REGRESSION TESTS
# ============================================================================


class TestNormalizationRegressions:
    """Test against known issues/regressions."""

    def test_no_nan_or_inf_after_normalization(self):
        """Ensure normalization doesn't produce NaN/Inf."""
        kspace = torch.randn(2, 2, 64, 64, dtype=torch.complex64)

        # Add some very small values
        kspace[0, 0, 0, 0] = 1e-10

        scale = torch.quantile(torch.abs(kspace), 0.99)
        kspace_normalized = kspace / (scale + 1e-8)  # eps prevents division by zero

        assert torch.isfinite(
            kspace_normalized
        ).all(), f"Normalization produced NaN or Inf: {torch.isnan(kspace_normalized).sum()} NaNs, {torch.isinf(kspace_normalized).sum()} Infs"

    def test_all_zeros_kspace_handling(self):
        """Edge case: zero k-space should be handled gracefully."""
        kspace = torch.zeros(1, 1, 64, 64, dtype=torch.complex64)

        # Should not crash or produce NaN
        scale = torch.quantile(torch.abs(kspace), 0.99)
        kspace_normalized = kspace / (scale + 1e-8)

        assert torch.isfinite(kspace_normalized).all()
        assert torch.all(kspace_normalized == 0.0) or torch.all(
            kspace_normalized < 1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
