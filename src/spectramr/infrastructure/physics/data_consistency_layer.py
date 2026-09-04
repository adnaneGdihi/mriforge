"""Data Consistency Layer for MRI Reconstruction.
==============================================

This module implements explicit Data Consistency (DC) layers that enforce
fidelity to measured k-space data within the network.
"""

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class DataConsistencyLayer(nn.Module):
    """
    Physics-Informed Data Consistency Layer.

    Enforces that the reconstructed image's k-space matches the measured
    k-space data at sampled frequencies.

    Operation:
        x_out = iFFT( (1 - M) * FFT(x_in) + M * y )

    where:
        x_in: Input image guess
        M: Sampling mask (1 at sampled locs, 0 elsewhere)
        y: Measured k-space (0 at unsampled locs)
    """

    def __init__(self, noise_robust: bool = False):
        """
        Args:
            noise_robust: If True, uses a softer consistency (e.g. weighted average)
                          Currently strictly enforcing hard consistency as per standard DC.
        """
        super().__init__()
        self.noise_robust = noise_robust

    def forward(
        self, image: torch.Tensor, measured_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Data Consistency.

        Args:
            image: Input image guess [B, C, H, W] (Complex-valued or 2-channel real)
            measured_kspace: Measured k-space [B, C, H, W] (Same format as image)
            mask: Sampling mask [B, 1, H, W] or broadcastable

        Returns:
            Corrected image [B, C, H, W]

        forward method for DataConsistencyLayer.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Transform Image -> K-Space
        # Handle complex vs stacked-real input
        is_complex = torch.is_complex(image)

        if is_complex:
            k_guessed = fft2c(image)
        else:
            # Assume 2-channel real represents complex
            # We need to treat it as complex for FFT operations
            # Check for interleaved real/imag channels (e.g., [B, 2*Coils, H, W])
            if image.shape[1] % 2 == 0:
                img_complex = torch.complex(image[:, 0::2, ...], image[:, 1::2, ...])
                k_guessed_complex = fft2c(img_complex)
                # Keep distinct for now
            else:
                # If C is odd, we can't easily do DC
                # unless we have measured k-space for those features (unlikely)
                # or if we treat channels as independent modification (Batchelor et al).
                # For standard MRI DC, we usually operate on the "image" channels.
                # If this layer is used inside a network with C > 2 hidden channels,
                # we technically shouldn't apply DC unless those channels map to image space.
                # Assuming this is used at "image-like" stages or C=2.
                # If C > 2, we might verify if it's pairs.
                raise ValueError(
                    f"DataConsistencyLayer expects 2-channel (Re/Im) or Complex input, got {image.shape}"
                )

        # Ensure mask is float for arithmetic (bool tensors don't support subtraction)
        if mask.dtype == torch.bool:
            mask = mask.float()

        # Mask broadcastability guard. The k-space sampling mask is
        # *coil-independent* by physics — the same Cartesian / radial
        # pattern is acquired across every coil. Some upstream paths
        # (e.g. data_consistency_layer.py callers via ``unrolled_*``
        # generators) can hand the layer a mask shaped like the input
        # complex tensor instead of [B, 1, H, W]. The 2026-05-10 cluster
        # rerun saw shapes like k=[B, 8, H, W] with mask=[B, 4, H, W]
        # which fails non-singleton-dim broadcast at line ``k * (1-m) +
        # measured * m`` below.
        #
        # Resolution: collapse to a 1-channel mask when channel counts
        # disagree. We use ``amax`` (logical OR over coil dim) so that
        # a frequency is treated as "sampled" if ANY coil sampled it.
        # This matches how multi-coil acquisitions actually work.
        target_channels = k_guessed_complex.shape[1] if not is_complex else k_guessed.shape[1]
        if mask.dim() >= 2 and mask.shape[1] != target_channels and mask.shape[1] != 1:
            mask = mask.amax(dim=1, keepdim=True)

        # 2. Apply Consistency
        # For complex tensor:
        if is_complex:
            # Hard DC: Replace sampled freq with measured
            # k_out = k_guessed * (1 - mask) + measured_kspace * mask
            # Note: measured_kspace should already be masked (0 at unsampled), but * mask is safer.
            k_corrected = k_guessed * (1.0 - mask) + measured_kspace * mask

            # 3. Transform K-Space -> Image
            image_corrected = ifft2c(k_corrected)
            return image_corrected

        else:
            # For 2-channel real representation
            # We need measured_kspace to also be 2-channel real or convertable
            if torch.is_complex(measured_kspace):
                # Unlikely if image is real-stacked, but handle it
                measured_kspace_complex = measured_kspace
            else:
                measured_kspace_complex = torch.complex(
                    measured_kspace[:, 0::2, ...], measured_kspace[:, 1::2, ...]
                )

            k_guessed_shape = k_guessed_complex.shape
            measured_shape = measured_kspace_complex.shape
            mask_shape = mask.shape
            try:
                k_corrected_complex = (
                    k_guessed_complex * (1.0 - mask) + measured_kspace_complex * mask
                )
            except Exception as e:
                logger.error(
                    "[DC LAYER CRASH] k_guessed_complex: %s, measured_kspace_complex: %s, mask: %s",
                    k_guessed_shape,
                    measured_shape,
                    mask_shape,
                )
                raise e

            # iFFT
            image_corrected_complex = ifft2c(k_corrected_complex)

            # Stack back to Real/Imag keeping the interleaved channel layout
            return torch.stack(
                [image_corrected_complex.real, image_corrected_complex.imag], dim=2
            ).flatten(1, 2)
