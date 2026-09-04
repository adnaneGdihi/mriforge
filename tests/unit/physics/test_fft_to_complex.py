"""Unit tests for FFT operations with complex tensor handling.

Tests the _to_complex function to ensure correct handling of:
- [B, 2, H, W] real/imaginary channel format
- Complex tensors (pass-through)
- Last-dimension encoding format [*, *, *, 2]
- Pure real tensors (cast to complex with zero phase)
"""

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import (
    _to_complex,
    coil_combine_rss,
    fft2c,
    ifft2c,
)


class TestToComplex:
    """Test _to_complex function with various input formats."""

    def test_to_complex_channel_format_preserve_shape(self):
        """Test [B, 2, H, W] format preserves shape and adds channel dim.

        This is the critical test case for the validation metrics bug fix.
        Input [B, 2, H, W] where channels are real/imag should become [B, 1, H, W] complex.
        """
        # Input: [B=18, C=2 real/imag, H=256, W=256]
        x = torch.randn(18, 2, 256, 256)
        result = _to_complex(x)

        # Expected: [B=18, C_coils=1, H=256, W=256] complex64
        assert result.shape == torch.Size([18, 1, 256, 256])
        assert torch.is_complex(result)
        assert result.dtype == torch.complex64

    def test_to_complex_channel_format_values(self):
        """Test that real/imag channels are correctly combined.

        Uses shape [1, 2, 3, 4] to avoid ambiguity with the last-dim==2
        (view_as_complex) path. When last dim != 2, _to_complex uses the
        channel interleaving path ([B, 2, H, W] -> [B, 1, H, W] complex).
        """
        # Create known values with unambiguous spatial dims
        real_part = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]]
        )  # [1, 1, 3, 4]
        imag_part = torch.tensor(
            [
                [
                    [
                        [13.0, 14.0, 15.0, 16.0],
                        [17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0],
                    ]
                ]
            ]
        )  # [1, 1, 3, 4]

        # Stack as [B, 2, H, W]
        x = torch.cat([real_part, imag_part], dim=1)  # [1, 2, 3, 4]

        result = _to_complex(x)

        # Verify values - should be [1, 1, 3, 4] complex
        expected_complex = torch.complex(real_part, imag_part)
        assert (
            result.shape == expected_complex.shape
        ), f"Shape mismatch: {result.shape} vs {expected_complex.shape}"
        assert torch.allclose(result, expected_complex)

    def test_to_complex_already_complex_passthrough(self):
        """Test that already-complex tensors pass through unchanged."""
        x = torch.randn(18, 1, 256, 256, dtype=torch.complex64)
        result = _to_complex(x)

        assert result.shape == x.shape
        assert torch.allclose(result, x)
        assert result.dtype == torch.complex64

    def test_to_complex_last_dim_encoding(self):
        """Test (..., 2) format where last dim encodes real/imag."""
        # Shape [B, H, W, 2] - last dimension encodes real/imag
        x = torch.randn(18, 256, 256, 2)
        result = _to_complex(x)

        # Should use view_as_complex
        assert torch.is_complex(result)
        assert result.shape == torch.Size([18, 256, 256])

    def test_to_complex_pure_real_tensor(self):
        """Test pure real tensor casting to complex with zero phase."""
        # Arbitrary shape real tensor
        x = torch.randn(18, 3, 256, 256)
        result = _to_complex(x)

        # Should be complex with zero imaginary part
        assert torch.is_complex(result)
        assert result.dtype == torch.complex64
        assert torch.allclose(result.imag, torch.zeros_like(result.imag))
        assert torch.allclose(result.real, x)

    def test_to_complex_1d_tensor(self):
        """Test 1D tensor handling."""
        x = torch.randn(256)
        result = _to_complex(x)

        # Should cast to complex with zero phase
        assert torch.is_complex(result)
        assert torch.allclose(result.real, x)


class TestCompleteValidationPipeline:
    """Test the complete pipeline used in validation metrics.

    The validation metrics pipeline is:
    [B,2,H,W] k-space → _to_complex → ifft2c → abs → RSS → metrics
    """

    def test_kspace_to_image_pipeline(self):
        """Test complete k-space to image transformation."""
        # Simulate k-space data: [B=18, C=2 real/imag, H=256, W=256]
        kspace = torch.randn(18, 2, 256, 256)

        # Step 1: Convert to complex [B, 1, 256, 256]
        kspace_complex = _to_complex(kspace)
        assert kspace_complex.shape == torch.Size([18, 1, 256, 256])
        assert torch.is_complex(kspace_complex)

        # Step 2: Apply inverse FFT
        image_complex = ifft2c(kspace_complex)
        assert image_complex.shape == torch.Size([18, 1, 256, 256])
        assert torch.is_complex(image_complex)

        # Step 3: Take magnitude
        image_mag = torch.abs(image_complex)
        assert image_mag.shape == torch.Size([18, 1, 256, 256])
        assert image_mag.dtype == torch.float32

        # Step 4: Apply RSS coil combination
        image_rss = coil_combine_rss(image_mag)
        assert image_rss.shape == torch.Size([18, 1, 256, 256])
        assert image_rss.dtype == torch.float32

        # Final shape should be ready for metrics computation
        assert not (
            image_rss.shape[-1] == 256 and image_rss.dim() == 3
        ), "Should be 4D, not 3D (bug would produce [18, 1, 256] losing width)"

    def test_kspace_to_image_roundtrip(self):
        """Test FFT roundtrip for real image.

        For a real image:
        1. FFT to k-space
        2. IFFT back to image
        3. Result should match original (up to numerical precision)

        Note: We compare magnitudes to handle phase ambiguity in roundtrip.
        """
        # Create real image
        real_image = torch.randn(1, 1, 64, 64)

        # Create k-space representation
        kspace_complex = fft2c(real_image)
        assert kspace_complex.shape == torch.Size([1, 1, 64, 64])

        # Reconstruct image
        reconstructed = ifft2c(kspace_complex)
        assert reconstructed.shape == torch.Size([1, 1, 64, 64])

        # Get magnitude (handles phase ambiguity)
        reconstructed_mag = torch.abs(reconstructed)
        original_mag = torch.abs(real_image)

        # Should match original (up to FFT roundtrip precision)
        assert torch.allclose(original_mag, reconstructed_mag, atol=1e-5)

    def test_metrics_compatible_shapes(self):
        """Verify output shapes are compatible with metric computation.

        Metrics require:
        - 4D tensors [B, C, H, W]
        - C=1 for single-image metrics
        - C can be >1 for multi-coil comparison
        """
        # Simulate batch of k-space data
        batch_size = 18
        height, width = 256, 256

        # Prediction and target in k-space
        pred_kspace = torch.randn(batch_size, 2, height, width)
        target_kspace = torch.randn(batch_size, 2, height, width)

        # Transform to image domain (as in validation)
        pred_image = torch.abs(ifft2c(_to_complex(pred_kspace)))
        target_image = torch.abs(ifft2c(_to_complex(target_kspace)))

        # Apply RSS
        pred_final = coil_combine_rss(pred_image)
        target_final = coil_combine_rss(target_image)

        # Check metric-compatible shapes
        assert pred_final.dim() == 4, "Metrics expect 4D tensors"
        assert target_final.dim() == 4, "Metrics expect 4D tensors"
        assert pred_final.shape == target_final.shape, "Shapes must match"
        assert pred_final.shape == torch.Size([batch_size, 1, height, width])
        assert pred_final.dtype == torch.float32
        assert target_final.dtype == torch.float32


class TestBugFix:
    """Regression tests for the validation metrics shape bug.

    Bug: [18, 2, 256, 256] → [18, 1, 256] (missing width)
    Expected: [18, 2, 256, 256] → [18, 1, 256, 256]
    """

    def test_shape_18_2_256_256_not_collapsed_to_3d(self):
        """The core bug: shape should NOT collapse to [18, 1, 256]."""
        x = torch.randn(18, 2, 256, 256)
        result = _to_complex(x)

        # MUST be 4D, not 3D
        assert result.dim() == 4, f"Expected 4D, got {result.dim()}D: {result.shape}"
        assert result.shape == torch.Size([18, 1, 256, 256])

        # MUST have width dimension = 256
        assert result.shape[-1] == 256, f"Width should be 256, got {result.shape[-1]}"
        assert result.shape[-2] == 256, f"Height should be 256, got {result.shape[-2]}"

    def test_complete_pipeline_shape_preservation(self):
        """Full pipeline should not lose spatial dimensions."""
        x = torch.randn(18, 2, 256, 256)

        # Full pipeline
        x_complex = _to_complex(x)
        x_ifft = ifft2c(x_complex)
        x_mag = torch.abs(x_ifft)
        x_rss = coil_combine_rss(x_mag)

        # Every step should preserve spatial dimensions
        assert x_complex.shape[-2:] == torch.Size([256, 256])
        assert x_ifft.shape[-2:] == torch.Size([256, 256])
        assert x_mag.shape[-2:] == torch.Size([256, 256])
        assert x_rss.shape[-2:] == torch.Size([256, 256])

        # Final shape should be 4D
        assert x_rss.dim() == 4
        assert x_rss.shape == torch.Size([18, 1, 256, 256])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
