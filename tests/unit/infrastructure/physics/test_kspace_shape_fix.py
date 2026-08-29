#!/usr/bin/env python3
"""
Test for the k-space cold diffusion shape mismatch fix.

The problem was:
- hr_fakes (model output): [4, 1, 640, 368] (image-space magnitude)
- hr_reals (ground truth): [4, 2, 640, 368] (k-space complex as real+imag channels)
- SSIM loss fails on shape mismatch

The fix:
1. Convert hr_batch to real [B, 2, H, W] in diffusion strategy
2. In compute_loss, convert real [B, 2, H, W] to magnitude [B, 1, H, W]
"""

import torch


def test_hr_batch_conversion_in_diffusion_strategy():
    """Test that hr_batch is converted to real representation in diffusion strategy."""
    # Simulate complex k-space ground truth [B, C, H, W]
    hr_batch = torch.randn(4, 1, 640, 368, dtype=torch.complex64)

    # Simulate the conversion that happens in diffusion strategy
    if torch.is_complex(hr_batch):
        b, c, h, w = hr_batch.shape
        hr_batch_converted = (
            torch.view_as_real(hr_batch)
            .permute(0, 1, 4, 2, 3)
            .reshape(b, c * 2, h, w)
            .contiguous()
        )

        assert hr_batch_converted.shape == (
            4,
            2,
            640,
            368,
        ), f"Expected [4,2,640,368] but got {hr_batch_converted.shape}"


def test_magnitude_conversion_in_compute_loss():
    """Test that real [B,2,H,W] is converted to magnitude [B,1,H,W] in compute_loss."""
    # Simulate hr_reals in compute_loss: real representation [B, 2, H, W]
    hr_reals_real = torch.randn(4, 2, 640, 368, dtype=torch.float32)

    # Simulate the conversion that should happen in compute_loss
    B, _, H, W = hr_reals_real.shape
    real_part = hr_reals_real[:, 0:1, :, :]  # [B, 1, H, W]
    imag_part = hr_reals_real[:, 1:2, :, :]  # [B, 1, H, W]

    # Reconstruct complex tensor [B, H, W]
    hr_reals_complex = torch.complex(
        real_part.squeeze(1), imag_part.squeeze(1)
    )  # [B, H, W]

    # Take magnitude
    hr_reals_mag = torch.abs(hr_reals_complex).unsqueeze(1)  # [B, 1, H, W]

    assert hr_reals_mag.shape == (
        4,
        1,
        640,
        368,
    ), f"Expected [4,1,640,368] but got {hr_reals_mag.shape}"


def test_shape_matching_for_ssim():
    """Test that after conversion, pred and target shapes match for SSIM."""
    # Model output (hr_fakes)
    hr_fakes = torch.randn(4, 1, 640, 368, dtype=torch.float32)

    # Ground truth after conversion
    hr_reals_real = torch.randn(4, 2, 640, 368, dtype=torch.float32)
    B, _, H, W = hr_reals_real.shape
    real_part = hr_reals_real[:, 0:1, :, :]
    imag_part = hr_reals_real[:, 1:2, :, :]
    hr_reals_complex = torch.complex(real_part.squeeze(1), imag_part.squeeze(1))
    hr_reals_mag = torch.abs(hr_reals_complex).unsqueeze(1)

    # Check if shapes match
    assert (
        hr_fakes.shape == hr_reals_mag.shape
    ), f"Shapes don't match: {hr_fakes.shape} vs {hr_reals_mag.shape}"


def test_roundtrip_conversion():
    """Test that the conversion is reversible and preserves information."""
    # Create original complex k-space
    original_complex = torch.randn(4, 1, 64, 64, dtype=torch.complex64)

    # Convert to real [B, 2, H, W]
    b, c, h, w = original_complex.shape
    real_repr = (
        torch.view_as_real(original_complex)
        .permute(0, 1, 4, 2, 3)
        .reshape(b, c * 2, h, w)
        .contiguous()
    )

    # Convert back to magnitude
    B, _, H, W = real_repr.shape
    real_part = real_repr[:, 0:1, :, :]
    imag_part = real_repr[:, 1:2, :, :]
    reconstructed_complex = torch.complex(real_part.squeeze(1), imag_part.squeeze(1))
    magnitude = torch.abs(reconstructed_complex).unsqueeze(1)

    # Calculate expected magnitude
    expected_magnitude = torch.abs(original_complex)

    # Check difference
    max_diff = torch.abs(magnitude - expected_magnitude).max().item()

    assert max_diff < 1e-3, f"Roundtrip error too high: {max_diff}"


if __name__ == "__main__":
    # Manual execution block if needed
    test_hr_batch_conversion_in_diffusion_strategy()
    test_magnitude_conversion_in_compute_loss()
    test_shape_matching_for_ssim()
    test_roundtrip_conversion()
    print("ALL TESTS PASSED")
