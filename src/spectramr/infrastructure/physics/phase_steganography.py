"""Phase-Domain Steganographic Marker for Motion Estimation.

Injects a mathematically pure synthetic sinusoidal grid **exclusively into
the complex phase** of the MRI data.  Because radiologists read magnitude
images, the marker is inherently invisible in the final clinical output
(``torch.abs(reconstructed)`` erases it by definition).

Physics:
    .. math::

        I_{marked}(x,y) = |I_{clean}| \\cdot
            \\exp\\!\\bigl(j\\,[\\phi_{natural} +
            \\alpha \\sin(2\\pi f_x x)\\sin(2\\pi f_y y)]\\bigr)

    The spatial frequencies :math:`(f_x, f_y)` are chosen so the marker
    is spectrally disjoint from broadband anatomical content.

Erasure:
    ``torch.abs(marked_image)`` — zero computational cost.

Strain Extraction:
    After non-rigid deformation warps the phase grid, the local phase
    gradient :math:`\\nabla \\phi` encodes the full 2-D strain tensor,
    enabling dense internal deformation tracking without any physical marker.

Reference:
    Pelc et al., "Phase contrast magnetic resonance imaging,"
    *Magnetic Resonance Quarterly*, 1991.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PhaseSteganographicMarker(nn.Module):
    """Inject / extract a synthetic sinusoidal grid in the phase domain.

    Args:
        im_size: Image dimensions ``(H, W)``.
        spatial_freq_x: Marker frequency along x (cycles per FOV).
        spatial_freq_y: Marker frequency along y (cycles per FOV).
        amplitude: Peak phase modulation depth (radians).  :math:`\\pi`
            gives a full ±π grid; 0.5 is a gentle watermark.
        learnable: If ``True``, frequency and amplitude become trainable.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        spatial_freq_x: float = 8.0,
        spatial_freq_y: float = 8.0,
        amplitude: float = 0.5,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.im_size = im_size

        # Build normalised coordinate grids in [-1, 1]
        H, W = im_size
        y = torch.linspace(-1.0, 1.0, H)
        x = torch.linspace(-1.0, 1.0, W)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("_Y", Y)
        self.register_buffer("_X", X)

        freq_x = torch.tensor(spatial_freq_x)
        freq_y = torch.tensor(spatial_freq_y)
        amp = torch.tensor(amplitude)

        if learnable:
            self.freq_x = nn.Parameter(freq_x)
            self.freq_y = nn.Parameter(freq_y)
            self.amplitude = nn.Parameter(amp)
        else:
            self.register_buffer("freq_x", freq_x)
            self.register_buffer("freq_y", freq_y)
            self.register_buffer("amplitude", amp)

        logger.info(
            "[PhaseSteganographicMarker] im_size=%s, freq=(%.1f, %.1f), amp=%.2f rad, learnable=%s",
            im_size,
            spatial_freq_x,
            spatial_freq_y,
            amplitude,
            learnable,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _phase_grid(self) -> torch.Tensor:
        """Compute the raw phase modulation pattern ``[H, W]``."""
        return (
            self.amplitude
            * torch.sin(2 * math.pi * self.freq_x * self._X)
            * torch.sin(2 * math.pi * self.freq_y * self._Y)
        )

    def inject(self, clean_complex: torch.Tensor) -> torch.Tensor:
        """Embed the phase marker into a complex-valued image.

        Args:
            clean_complex: Complex tensor ``[B, C, H, W]`` (``torch.complex64``
                or 2-channel stacked ``[B, 2C, H, W]``).

        Returns:
            Marked complex tensor with same shape and dtype.
        """
        is_stacked = not clean_complex.is_complex()
        if is_stacked:
            # Convert 2-channel stacked → complex
            B, C2, H, W = clean_complex.shape
            assert C2 % 2 == 0, f"Stacked tensor must have even channels, got {C2}"
            C = C2 // 2
            real = clean_complex[:, :C]
            imag = clean_complex[:, C:]
            cplx = torch.complex(real, imag)
        else:
            cplx = clean_complex

        magnitude = cplx.abs()
        natural_phase = cplx.angle()

        grid = self._phase_grid().to(cplx.device)  # [H, W]
        marked_phase = natural_phase + grid

        marked = magnitude * torch.exp(1j * marked_phase)

        if is_stacked:
            return torch.cat([marked.real, marked.imag], dim=1)
        return marked

    def erase(self, marked_image: torch.Tensor) -> torch.Tensor:
        """Remove the marker — simply compute magnitude.

        Args:
            marked_image: Complex or stacked tensor.

        Returns:
            Real-valued magnitude image ``[B, C, H, W]``.
        """
        if marked_image.is_complex():
            return marked_image.abs()
        # Stacked: first half real, second half imag
        B, C2, H, W = marked_image.shape
        C = C2 // 2
        real = marked_image[:, :C]
        imag = marked_image[:, C:]
        return torch.sqrt(real**2 + imag**2 + 1e-12)

    def extract_strain(
        self,
        corrupted_complex: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the 2-D strain tensor from phase-grid deformation.

        Computes spatial phase gradients and compares against the known
        ideal grid to recover the local deformation.

        Args:
            corrupted_complex: Complex tensor ``[B, C, H, W]`` after
                non-rigid motion has warped the phase grid.

        Returns:
            Strain tensor ``[B, 2, H, W]`` — approximate (∂u/∂x, ∂u/∂y)
            displacement gradients.
        """
        if not corrupted_complex.is_complex():
            B, C2, H, W = corrupted_complex.shape
            C = C2 // 2
            real = corrupted_complex[:, :C]
            imag = corrupted_complex[:, C:]
            cplx = torch.complex(real, imag)
        else:
            cplx = corrupted_complex

        # Extract measured phase and subtract ideal grid
        measured_phase = cplx.angle()  # [B, C, H, W]
        ideal_grid = self._phase_grid().to(cplx.device)  # [H, W]
        residual_phase = measured_phase - ideal_grid  # deviation from ideal

        # Spatial gradients of residual phase ≈ displacement gradient
        # Use central differences
        grad_x = (residual_phase[:, :, :, 2:] - residual_phase[:, :, :, :-2]) / 2.0
        grad_y = (residual_phase[:, :, 2:, :] - residual_phase[:, :, :-2, :]) / 2.0

        # Pad to original size
        grad_x = torch.nn.functional.pad(grad_x, (1, 1, 0, 0), mode="replicate")
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 1, 1), mode="replicate")

        # Average across channels
        grad_x = grad_x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        grad_y = grad_y.mean(dim=1, keepdim=True)

        return torch.cat([grad_x, grad_y], dim=1)  # [B, 2, H, W]

    def forward(self, clean_complex: torch.Tensor) -> torch.Tensor:
        """Forward pass — injects the phase marker.

        Alias for :meth:`inject` for ``nn.Module`` compatibility.

        forward method for PhaseSteganographicMarker.

        Executes PyTorch tensor operations.

        Args:
            clean_complex (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.inject(clean_complex)


__all__ = ["PhaseSteganographicMarker"]
