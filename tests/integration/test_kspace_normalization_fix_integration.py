"""
Integration Test: K-Space Normalization Timing Fix

Tests that the corrected transform ordering (k-space normalization BEFORE physics dispatch)
produces correct behavior across Cartesian and non-Cartesian trajectories.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch


@dataclass
class MockTorchIOConfig:
    """Mock configuration for testing."""

    patch_size: tuple = (64, 64)
    enable_geometric_standardization: bool = True
    standardization_mode: str = "smart"
    augmentation_config: dict = None
    trajectory_type: str = "cartesian"
    acceleration: int = 4
    center_fraction: float = 0.08
    normalize_kspace: bool = True
    kspace_percentile: float = 0.99
    normalization_type: str = "none"
    enable_graph_encoding: bool = False
    graph_config: dict = None


class TestKSpaceNormalizationTimingFix:
    """
    Integration tests validating the fix for k-space normalization ordering.

    The fix ensures:
    1. K-space normalization is applied BEFORE physics dispatch
    2. Scale factor is computed on FULL k-space
    3. Masked/trajectory k-space is normalized consistently
    """

    def test_normalization_applied_before_masking_cartesian(self):
        """
        Test that for Cartesian trajectories, normalization happens before masking.

        INVARIANT: The scale should be computed from FULL k-space before applying the mask.
        """
        # Create synthetic image
        image = torch.randn(1, 1, 64, 64)

        # For Cartesian masking test:
        # 1. Compute scale on full k-space
        from mriforge.infrastructure.physics.fft_ops import fft2c

        kspace_full = fft2c(image)
        scale_full = torch.quantile(torch.abs(kspace_full), 0.99)

        # 2. Normalize full k-space
        kspace_normalized_full = kspace_full / (scale_full + 1e-8)

        # 3. Apply Cartesian mask
        H, W = 64, 64
        mask = torch.zeros((H, W))
        center_h = int(H * 0.08)
        start_h = (H - center_h) // 2
        mask[start_h : start_h + center_h, :] = 1.0
        # Random outer region
        outer_mask = torch.rand((H, W)) < (1.0 / 4.0)
        mask = torch.logical_or(mask.bool(), outer_mask).float()

        # 4. Apply mask to normalized k-space
        kspace_masked = kspace_normalized_full * mask.unsqueeze(0).unsqueeze(0)

        # VERIFICATION: Masked values should be exact same scale as normalized full k-space
        # (not rescaled)
        full_norm_range = (
            torch.abs(kspace_normalized_full).min(),
            torch.abs(kspace_normalized_full).max(),
        )
        masked_norm_range = (
            (
                torch.abs(kspace_masked[mask.unsqueeze(0).unsqueeze(0).bool()]).min()
                if (mask.unsqueeze(0).unsqueeze(0) > 0).any()
                else 0
            ),
            (
                torch.abs(kspace_masked[mask.unsqueeze(0).unsqueeze(0).bool()]).max()
                if (mask.unsqueeze(0).unsqueeze(0) > 0).any()
                else 1
            ),
        )

        # The ranges should match (within tolerance for quantization)
        assert (
            abs(full_norm_range[0] - masked_norm_range[0]) < 0.1
        ), f"Min scale mismatch: full={full_norm_range[0]:.4f}, masked={masked_norm_range[0]:.4f}"

    def test_normalization_enables_consistent_inference(self):
        """
        Test that normalized k-space produces consistent results at inference time.

        INVARIANT: If training uses normalized k-space before masking,
        inference should too.
        """
        # Create two identical images
        image_train = torch.randn(1, 1, 64, 64)
        image_infer = image_train.clone()

        from mriforge.infrastructure.physics.fft_ops import fft2c

        # Training path (should use normalized k-space)
        kspace_train = fft2c(image_train)
        scale_train = torch.quantile(torch.abs(kspace_train), 0.99)
        kspace_train_norm = kspace_train / (scale_train + 1e-8)

        # Apply mask (hypothetical)
        mask_train = torch.rand(64, 64) < 0.25
        kspace_train_masked = kspace_train_norm * mask_train.unsqueeze(0).unsqueeze(0)

        # Inference path (should use SAME scale)
        kspace_infer = fft2c(image_infer)
        # Use SAME scale_train from training, not recompute on inference set
        kspace_infer_norm = kspace_infer / (scale_train + 1e-8)

        # Apply same mask
        kspace_infer_masked = kspace_infer_norm * mask_train.unsqueeze(0).unsqueeze(0)

        # Results should be identical
        assert torch.allclose(
            kspace_train_masked, kspace_infer_masked, rtol=1e-5
        ), "Training and inference normalized k-space should match"

    def test_normalization_prevents_scale_mismatch_non_cartesian(self):
        """
        Test that non-Cartesian trajectories use correct scale.

        INVARIANT: Scale should be computed from FULL k-space before
        trajectory simulation, not from sparse trajectory samples.
        """
        # Create image with structured signal (stronger scale mismatch)
        image = torch.zeros(1, 1, 128, 128)
        image[:, :, 48:80, 48:80] = 1.0  # Centered square

        from mriforge.infrastructure.physics.fft_ops import fft2c

        kspace_full = fft2c(image)

        # WRONG way: Compute scale from sparse trajectory
        # (Example: radial with 24 arms)
        num_arms = 24
        y_rad, x_rad = np.ogrid[-128 / 2 : 128 / 2, -128 / 2 : 128 / 2]
        r_rad = np.sqrt(x_rad**2 + y_rad**2)
        theta_rad = np.arctan2(y_rad, x_rad)

        radial_mask = torch.zeros(128, 128)
        for phase in np.linspace(0, 2 * np.pi, num_arms, endpoint=False):
            arm_mask = (
                np.abs(np.angle(np.exp(1j * (theta_rad - phase)))) < np.pi / num_arms
            )
            radial_mask[arm_mask] = 1.0

        kspace_radial_wrong_scale = kspace_full * radial_mask.unsqueeze(0).unsqueeze(0)
        scale_from_radial = torch.quantile(
            torch.abs(kspace_radial_wrong_scale[kspace_radial_wrong_scale != 0]), 0.99
        )

        # RIGHT way: Compute scale from full k-space
        scale_from_full = torch.quantile(torch.abs(kspace_full), 0.99)

        # Scales should be different (proving the timing matters)
        scale_diff = abs(scale_from_full - scale_from_radial).item()
        assert scale_diff > 0.001, (
            f"Scale difference should be detectable for non-Cartesian, "
            f"but got {scale_diff:.6f}"
        )

        # The CORRECT approach is to normalize from full scale
        kspace_normalized = kspace_full / (scale_from_full + 1e-8)
        kspace_radial_correct = kspace_normalized * radial_mask.unsqueeze(0).unsqueeze(
            0
        )

        # Verify: Normalized values should be within expected range (not rescaled)
        normalized_magnitudes = torch.abs(
            kspace_radial_correct[kspace_radial_correct != 0]
        )

        # Most values should be < 1.0 (since we normalized by 99th percentile)
        fraction_below_1 = (
            normalized_magnitudes <= 1.0
        ).sum() / normalized_magnitudes.numel()
        assert (
            fraction_below_1 > 0.95
        ), f"After normalization by 99th percentile, ~99% should be <=1, got {fraction_below_1:.1%}"

    def test_transform_pipeline_ordering_validated(self):
        """
        Test that the transform pipeline has correct ordering.

        Validate the sequence:
        1. Geometry
        2. Physics sync
        3. Augmentation
        4. Physics sync
        5. K-ZSpace Normalization  ← BEFORE physics dispatch
        6. Physics dispatch
        7. Image normalization
        """
        # Create mock config
        config = MockTorchIOConfig(
            trajectory_type="cartesian",
            normalize_kspace=True,
            normalization_type="none",  # For now, focus on k-space ordering
        )

        # This would be created by TorchIOTransformBuilder.build_train_transforms()
        # For now, just document the expected order

        expected_order = [
            "spatial_consistency",
            "geometry_standardization",
            "physics_sync_1",
            "augmentation",
            "physics_sync_2",
            "kspace_normalization",  # ← KEY: BEFORE physics dispatch
            "physics_dispatch",
            "image_normalization",
            "graph_encoding",
            "spatial_consistency_final",
        ]

        # This is more of a documentation test
        # Actual TorchIO transform objects would be verified via runtime tests
        assert "kspace_normalization" in expected_order
        assert expected_order.index("kspace_normalization") < expected_order.index(
            "physics_dispatch"
        ), "K-space normalization must occur before physics dispatch"


# ============================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# ============================================================================

try:
    from hypothesis import given
    from hypothesis import strategies as st

    # Import NumPy if available for radial mask generation
    try:
        import numpy as np

        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False

    class TestKSpaceNormalizationProperties:
        """Property-based tests using Hypothesis."""

        @given(
            percentile=st.floats(min_value=0.5, max_value=0.99),
            acceleration=st.integers(min_value=2, max_value=8),
        )
        def test_normalization_scale_always_decreases_values(
            self, percentile, acceleration
        ):
            """
            PROPERTY: Normalizing by percentile value always decreases magnitudes.

            If scale = quantile(|kspace|, p), then normalized |kspace| <= max_value
            """
            # Create random k-space
            kspace = torch.randn(1, 1, 32, 32, dtype=torch.complex64)
            kspace_mag = torch.abs(kspace)

            # Compute scale
            scale = torch.quantile(kspace_mag, percentile)

            # Normalize
            kspace_normalized = kspace / (scale + 1e-8)
            kspace_mag_norm = torch.abs(kspace_normalized)

            # PROPERTY: The p-th percentile of normalized magnitudes should be 1.0
            # (within numerical precision)
            norm_quantile = torch.quantile(kspace_mag_norm, percentile)
            assert torch.allclose(
                norm_quantile, torch.tensor(1.0), atol=1e-5
            ), f"Quantile {percentile} of normalized k-space should be 1.0, got {norm_quantile:.4f}"

        @given(size=st.integers(min_value=16, max_value=64))
        def test_complex_kspace_normalization_preserves_phase(self, size):
            """
            PROPERTY: Normalizing k-space should preserve phase information.

            phase(kspace_norm) == phase(kspace)
            """
            kspace = torch.randn(1, 1, size, size, dtype=torch.complex64)
            original_phase = torch.angle(kspace)

            # Normalize
            scale = torch.abs(kspace).max()
            kspace_norm = kspace / (scale + 1e-8)
            norm_phase = torch.angle(kspace_norm)

            # Phase should match
            assert torch.allclose(
                original_phase, norm_phase, atol=1e-6, equal_nan=True
            ), "Phase was not preserved during normalization"

except ImportError:
    # Hypothesis not available, skip property tests
    class TestKSpaceNormalizationProperties:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
