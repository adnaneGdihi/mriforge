"""Transform Operations Utilities

This module contains utilities for FFT operations, complex number handling,
and image reconstruction transformations used across training strategies.

NOTE: This module now uses the unified FFT operations from core.physics.fft_ops
to avoid duplication and ensure consistency.
"""

import torch

# Import unified FFT operations to avoid duplication
from spectramr.infrastructure.physics.fft_ops import (
    FFTTransformer,
    _to_complex,
    _to_ri,
    coil_combine_rss,
    fft2c,
    ifft2c,
)


class ComplexTensorHandler:
    """Utility class for handling complex tensor operations.

    This class provides methods for converting between different complex
    tensor representations and performing complex arithmetic.
    """

    @staticmethod
    def real_imag_to_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Convert real and imaginary parts to complex tensor.

        Args:
            real: Real part tensor
            imag: Imaginary part tensor

        Returns:
            Complex tensor

        """
        # Use unified complex conversion
        combined = torch.stack((real, imag), dim=-1)
        return _to_complex(combined)

    @staticmethod
    def complex_to_real_imag(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert complex tensor to real and imaginary parts.

        Args:
            x: Complex tensor

        Returns:
            Tuple of (real, imag) tensors

        """
        # Use unified real-imag conversion
        ri_tensor = _to_ri(x)
        return ri_tensor[..., 0], ri_tensor[..., 1]

    @staticmethod
    def magnitude(x: torch.Tensor) -> torch.Tensor:
        """Compute magnitude of complex tensor.

        Args:
            x: Complex tensor

        Returns:
            Magnitude tensor

        """
        x_complex = _to_complex(x)
        return torch.abs(x_complex)

    @staticmethod
    def phase(x: torch.Tensor) -> torch.Tensor:
        """Compute phase of complex tensor.

        Args:
            x: Complex tensor

        Returns:
            Phase tensor

        """
        x_complex = _to_complex(x)
        return torch.angle(x_complex)

    @staticmethod
    def combine_coil_data(coil_data: torch.Tensor, _coil_dim: int = 1) -> torch.Tensor:
        """Combine multi-coil data using RSS (Root Sum of Squares).

        Args:
            coil_data: Multi-coil data of shape [B, coils, H, W]
            coil_dim: Dimension containing coil data

        Returns:
            Combined single-channel data of shape [B, 1, H, W]

        """
        # Use unified coil combination
        return coil_combine_rss(coil_data)

    @staticmethod
    def split_coil_data(
        combined_data: torch.Tensor,
        num_coils: int,
        _coil_dim: int = 1,
    ) -> torch.Tensor:
        """Split single-channel data into multiple coils (uniform distribution).

        Args:
            combined_data: Single-channel data of shape [B, 1, H, W]
            num_coils: Number of coils to split into
            coil_dim: Dimension to place coil data

        Returns:
            Multi-coil data of shape [B, num_coils, H, W]

        """
        # Uniform split across coils
        split_data = combined_data.repeat(1, num_coils, 1, 1) / num_coils
        return split_data

    @staticmethod
    def tensor_to_complex(tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to complex format.

        Args:
            tensor: Input tensor

        Returns:
            Complex tensor

        """
        if tensor.shape[-1] == 2 and tensor.dim() == 5:
            # [B, coils, H, W, 2] -> complex
            return _to_complex(tensor)
        if tensor.shape[1] % 2 == 0 and tensor.dim() == 4:
            # [B, 2*C, H, W] -> [B, C, H, W] complex
            real = tensor[:, ::2]  # Even indices
            imag = tensor[:, 1::2]  # Odd indices
            return torch.complex(real, imag)
        # Assume already complex or single channel
        return tensor


class ReconstructionUtilities:
    """Utility class for image reconstruction operations.

    This class provides common reconstruction utilities used across
    different training strategies.
    """

    def __init__(self, device: torch.device | None = None):
        """Initialize reconstruction utilities.

        Args:
            device: Device for tensor operations

        """
        self.device = device or torch.device("cpu")
        self.fft_transformer = FFTTransformer(device)
        self.complex_handler = ComplexTensorHandler()

    def kspace_to_image(
        self,
        kspace: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Convert k-space data to image domain.

        Args:
            kspace: K-space data
            normalize: Whether to normalize the result

        Returns:
            Image domain data

        """
        # Handle complex format conversion
        if kspace.shape[-1] == 2 and kspace.dim() == 5:
            # [B, coils, H, W, 2] -> [B, coils, H, W]
            kspace = _to_complex(kspace)

        # Apply inverse FFT using unified operation
        image = ifft2c(kspace)

        # Combine coils if multi-coil
        if image.shape[1] > 1:
            image = coil_combine_rss(image)

        # Normalize if requested
        if normalize:
            image = self._normalize_image(image)

        return image

    def image_to_kspace(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image domain data to k-space.

        Args:
            image: Image domain data

        Returns:
            K-space data

        """
        # Use unified FFT operation
        return fft2c(image)

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range.

        Delegates to SSOT ``normalize_minmax`` from
        ``spectramr.data.transforms.normalization``.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        from spectramr.data.transforms.normalization import normalize_minmax

        normalized, _ = normalize_minmax(image, out_range=(0.0, 1.0))
        return normalized


def create_fft_transformer(device: torch.device | None = None) -> FFTTransformer:
    """Factory function for creating FFT transformer.

    NOTE: This now serves as a compatibility layer that uses
    the unified FFT operations from spectramr.infrastructure.physics.fft_ops.

    Args:
        device: Device for tensor operations

    Returns:
        Configured FFTTransformer instance

    """
    return FFTTransformer(device)


def create_complex_handler() -> ComplexTensorHandler:
    """Factory function for creating complex tensor handler.

    NOTE: This now serves as a compatibility layer that uses
    the unified complex operations from spectramr.infrastructure.physics.fft_ops.

    Returns:
        ComplexTensorHandler instance

    """
    return ComplexTensorHandler()


def create_reconstruction_utilities(
    device: torch.device | None = None,
) -> ReconstructionUtilities:
    """Factory function for creating reconstruction utilities.

    NOTE: This now serves as a compatibility layer that uses
    the unified operations from spectramr.infrastructure.physics.fft_ops.

    Args:
        device: Device for tensor operations

    Returns:
        Configured ReconstructionUtilities instance

    """
    return ReconstructionUtilities(device)
