"""SPAMM Grid Injection & Dirac Notch Erasure for Pipeline B.

Implements magnitude-domain Spatial Modulation of Magnetization (SPAMM)
tagging and its spectral erasure via Dirac notch filtering in k-space.

SPAMM Physics:
    A 1-1 SPAMM preparation creates a cosine modulation of the
    longitudinal magnetization:

    .. math::

        I_{tagged}(x, y) = I(x, y) \\cdot \\cos(2\\pi f_x x)
            \\cdot \\cos(2\\pi f_y y)

    In k-space, this multiplication becomes convolution with four Dirac
    deltas at :math:`(\\pm f_x, \\pm f_y)`.

Dirac Notch Filter:
    A binary mask that zeroes out the four spectral peaks of the SPAMM
    grid in k-space.  When applied to the DC loss of an MR-NeRF, the
    network is never penalised at tag frequencies and therefore
    mathematically cannot learn to render them.

Reference:
    Zerhouni et al., "Human Heart: Tagging with MR Imaging — A Method
    for Noninvasive Assessment of Myocardial Motion," *Radiology*, 1988.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class SPAMMGridInjector(nn.Module):
    """Inject a cosine tagging grid into image magnitude.

    Unlike Phase Steganography (M7), SPAMM modulates the **magnitude**
    directly — exactly as a real MR scanner does with a 1-1 SPAMM
    preparation pulse.

    Args:
        spatial_freq_x: Tag frequency in x (cycles per FOV).
        spatial_freq_y: Tag frequency in y (cycles per FOV).
        tag_contrast: Modulation depth ∈ [0, 1].  1.0 is full cosine
            (tags go to zero intensity).
    """

    def __init__(
        self,
        spatial_freq_x: float = 8.0,
        spatial_freq_y: float = 8.0,
        tag_contrast: float = 0.8,
    ) -> None:
        super().__init__()
        self.spatial_freq_x = spatial_freq_x
        self.spatial_freq_y = spatial_freq_y
        self.tag_contrast = tag_contrast

    def _tag_pattern(self, H: int, W: int, device: torch.device) -> Tensor:
        """Generate the cosine tag pattern ``[H, W]``."""
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        pattern = torch.cos(2 * math.pi * self.spatial_freq_x * X) * torch.cos(
            2 * math.pi * self.spatial_freq_y * Y
        )
        # Scale: 1.0 at peaks, (1 - contrast) at troughs
        return 1.0 - self.tag_contrast * (1.0 - pattern) / 2.0

    def inject(self, image: Tensor) -> Tensor:
        """Apply SPAMM tagging to image magnitude.

        Args:
            image: ``[B, C, H, W]`` — real or complex.

        Returns:
            Tagged image with same shape.
        """
        H, W = image.shape[-2:]
        pattern = self._tag_pattern(H, W, image.device)

        if image.is_complex():
            return image * pattern
        return image * pattern

    def forward(self, image: Tensor) -> Tensor:
        """Alias for :meth:`inject`.

        forward method for SPAMMGridInjector.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.inject(image)


class DiracNotchFilter(nn.Module):
    """Binary k-space mask zeroing SPAMM tag frequencies.

    Creates a mask with notches at the four Dirac delta positions
    corresponding to the SPAMM tag frequencies.

    Args:
        im_size: Image dimensions ``(H, W)``.
        spatial_freq_x: SPAMM frequency in x (must match injector).
        spatial_freq_y: SPAMM frequency in y (must match injector).
        notch_radius: Radius (in k-space pixels) of each notch.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        spatial_freq_x: float = 8.0,
        spatial_freq_y: float = 8.0,
        notch_radius: int = 3,
    ) -> None:
        super().__init__()
        H, W = im_size
        self.im_size = im_size

        # Compute Dirac delta positions in discrete k-space indices.
        # The injector applies cos(2π·f·X) on X ∈ [-1, 1] (FOV width 2), i.e.
        # 2·f full cycles across the FOV, so each tag peak sits 2·f bins from
        # the centre — not f. A notch at ±round(f) misses the tags entirely;
        # the offset must be ±round(2·f) to null the SPAMM harmonics.
        kx_offset = int(round(2.0 * spatial_freq_x))
        ky_offset = int(round(2.0 * spatial_freq_y))
        cx, cy = W // 2, H // 2

        # Four Dirac peaks: (±fx, ±fy)
        peaks = [
            (cy + ky_offset, cx + kx_offset),
            (cy + ky_offset, cx - kx_offset),
            (cy - ky_offset, cx + kx_offset),
            (cy - ky_offset, cx - kx_offset),
        ]

        # Build mask: 1 everywhere except notches
        mask = torch.ones(H, W)
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        for py, px in peaks:
            dist_sq = (yy - py).float() ** 2 + (xx - px).float() ** 2
            mask = mask * (dist_sq > notch_radius**2).float()

        self.register_buffer("mask", mask)

        logger.info(
            "[DiracNotchFilter] im_size=%s, peaks=%s, radius=%d, coverage=%.1f%%",
            im_size,
            peaks,
            notch_radius,
            mask.sum().item() / mask.numel() * 100,
        )

    def apply_mask(self, kspace: Tensor) -> Tensor:
        """Apply notch filter to k-space data.

        Args:
            kspace: ``[B, C, H, W]`` complex k-space.

        Returns:
            Notched k-space with SPAMM frequencies zeroed.
        """
        return kspace * self.mask

    def get_loss_mask(self) -> Tensor:
        """Return the binary mask for use in DC loss computation.

        Returns:
            ``[H, W]`` binary mask (1 = include in loss, 0 = notch).
        """
        return self.mask

    def forward(self, kspace: Tensor) -> Tensor:
        """Alias for :meth:`apply_mask`.

        forward method for DiracNotchFilter.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.apply_mask(kspace)


__all__ = ["DiracNotchFilter", "SPAMMGridInjector"]
