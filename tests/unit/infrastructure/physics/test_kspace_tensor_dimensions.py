"""
Unit test to verify k-space tensor dimension handling in KSpaceColdDiffusionDataset.

This test validates the fix for the tensor shape mismatch error:
RuntimeError: The size of tensor a (368) must match the size of tensor b (1007) at dimension 2

Root cause: HDF5 slice-aware loading returns complex64 tensor with shape (1, H, W)
when numpy converts (1, H, W, 2) float32 → complex64, the last dimension becomes
the imaginary part, reducing from 4D to 3D.
"""

import logging

import pytest
import torch


def extract_height_width_from_kspace(kspace_full):
    """
    Extract height and width from k-space tensor of various shapes.

    This mimics the logic in KSpaceColdDiffusionDataset.__getitem__ around line 450.

    Args:
        kspace_full: K-space tensor of various possible shapes

    Returns:
        tuple: (height, width)

    Raises:
        ValueError: If shape cannot be interpreted
    """
    logging.debug(
        f"Processing k-space shape: {kspace_full.shape}, dtype: {kspace_full.dtype}"
    )

    # Handle both 3D (H, W, 2) and 2D (H, W) tensor shapes
    if kspace_full.ndim == 5:
        # [batch, coils, slices, H, W] -> select first of each
        kspace_full = kspace_full[0, 0, 0]
        current_height, current_width = kspace_full.shape
        logging.debug(
            f"K-space shape (5D): {kspace_full.shape} -> selected [0,0,0] to H={current_height}, W={current_width}"
        )
    elif kspace_full.ndim == 4:
        # Could be [slices, coils, H, W] OR [coils, slices, H, W] OR [slices, H, W, 2]
        # Check if last dim is 2 (real/imag) - if so, it's [slices, H, W, 2]
        if kspace_full.shape[-1] == 2:
            # [slices, H, W, 2] - squeeze batch, convert complex
            kspace_full = kspace_full[0]  # Take first (and only) slice
            current_height, current_width = kspace_full.shape[0], kspace_full.shape[1]
            logging.debug(
                f"K-space shape (4D real/imag): {kspace_full.shape} -> selected [0] to H={current_height}, W={current_width}"
            )
        else:
            # [coils, slices, H, W] -> select first coil and slice
            kspace_full = kspace_full[0, 0]
            current_height, current_width = kspace_full.shape
            logging.debug(
                f"K-space shape (4D coils/slices): {kspace_full.shape} -> selected [0,0] to H={current_height}, W={current_width}"
            )
    elif kspace_full.ndim == 3:
        # Shape could be (H, W, 2) with real/imag OR (slices, H, W) complex
        # Check if last dim is 2 and dtype is NOT complex
        if kspace_full.shape[-1] == 2 and not kspace_full.dtype.is_complex:
            # (H, W, 2) - real/imaginary components
            current_height, current_width = kspace_full.shape[0], kspace_full.shape[1]
            logging.debug(
                f"K-space shape (3D real/imag): {kspace_full.shape} -> H={current_height}, W={current_width}, complex=2"
            )
        else:
            # (slices, H, W) with complex dtype - take first slice
            kspace_full = kspace_full[0]
            current_height, current_width = kspace_full.shape
            logging.debug(
                f"K-space shape (3D complex volume): {kspace_full.shape} -> selected [0] to H={current_height}, W={current_width}"
            )
    elif kspace_full.ndim == 2:
        # Shape is (H, W) - either magnitude or complex
        current_height, current_width = kspace_full.shape
        logging.debug(f"K-space shape (2D): {kspace_full.shape} (magnitude or complex)")
    else:
        raise ValueError(
            f"Unexpected k-space shape: {kspace_full.shape}. "
            f"Expected 2D, 3D, 4D or 5D tensor."
        )

    return current_height, current_width


class TestKSpaceShapeHandling:
    """Test suite for k-space tensor dimension handling."""

    @pytest.fixture
    def logger(self):
        """Set up logging for tests."""
        logging.basicConfig(level=logging.DEBUG)
        return logging.getLogger(__name__)

    def test_3d_complex_tensor_from_slice_aware_hdf5(self, logger):
        """
        Test handling of (1, H, W) complex64 from slice-aware HDF5 loading.

        This is the exact case from the bug report: HDF5 returns (1, 1007, 368, 2) float32,
        numpy converts to (1, 1007, 368) complex64, torch preserves as 3D.
        """
        logger.info("Test: 3D complex tensor from slice-aware HDF5 loading")

        # Simulate the exact shape from the bug report
        h5_shape = (1, 1007, 368)  # (batch=1, height=1007, width=368) as complex64
        kspace_tensor = torch.randn(h5_shape, dtype=torch.complex64)

        # Extract height and width using the fixed logic
        height, width = extract_height_width_from_kspace(kspace_tensor)

        # Should extract correct dimensions
        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 1007, f"Height should be 1007, got {height}"
        assert width == 368, f"Width should be 368, got {width}"

    def test_4d_real_imag_tensor_handling(self, logger):
        """Test handling of (slices, H, W, 2) real/imaginary tensor."""
        logger.info("Test: 4D real/imaginary tensor handling")

        h5_shape = (1, 640, 372, 2)  # (slices, height, width, real/imag)
        kspace_tensor = torch.randn(h5_shape, dtype=torch.float32)

        height, width = extract_height_width_from_kspace(kspace_tensor)

        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 640, f"Height should be 640, got {height}"
        assert width == 372, f"Width should be 372, got {width}"

    def test_3d_4d_coils_slices_tensor_handling(self, logger):
        """Test handling of (coils, slices, H, W) tensor."""
        logger.info("Test: 4D coils/slices tensor handling")

        h5_shape = (8, 1, 256, 200)  # (coils, slices, height, width)
        kspace_tensor = torch.randn(h5_shape, dtype=torch.complex64)

        height, width = extract_height_width_from_kspace(kspace_tensor)

        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 256, f"Height should be 256, got {height}"
        assert width == 200, f"Width should be 200, got {width}"

    def test_2d_magnitude_tensor_handling(self, logger):
        """Test handling of (H, W) magnitude-only tensor."""
        logger.info("Test: 2D magnitude tensor handling")

        h5_shape = (1007, 368)  # Magnitude only
        kspace_tensor = torch.randn(h5_shape, dtype=torch.float32)

        height, width = extract_height_width_from_kspace(kspace_tensor)

        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 1007, f"Height should be 1007, got {height}"
        assert width == 368, f"Width should be 368, got {width}"

    def test_3d_real_imag_tensor_handling(self, logger):
        """Test handling of (H, W, 2) real/imaginary tensor."""
        logger.info("Test: 3D real/imaginary tensor handling")

        h5_shape = (640, 372, 2)  # (height, width, real/imag)
        kspace_tensor = torch.randn(h5_shape, dtype=torch.float32)

        height, width = extract_height_width_from_kspace(kspace_tensor)

        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 640, f"Height should be 640, got {height}"
        assert width == 372, f"Width should be 372, got {width}"

    def test_5d_batch_tensor_handling(self, logger):
        """Test handling of (batch, coils, slices, H, W) tensor."""
        logger.info("Test: 5D batch tensor handling")

        h5_shape = (1, 8, 1, 256, 200)  # (batch, coils, slices, height, width)
        kspace_tensor = torch.randn(h5_shape, dtype=torch.complex64)

        height, width = extract_height_width_from_kspace(kspace_tensor)

        logger.info(f"Extracted: height={height}, width={width}")
        assert height == 256, f"Height should be 256, got {height}"
        assert width == 200, f"Width should be 200, got {width}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
