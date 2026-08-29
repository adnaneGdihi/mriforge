"""Extended Virtual Fiducial Operators for the 40-Part Framework.

Operators that complete the full operator chain specification:
- :class:`RFModulationOperator`       — B1+/B1- spatial block-diagonal modulation
- :class:`NUFFTTrajectoryCalibrator`  — gradient delay estimation + regridding
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# 1. RF Modulation Operator  (C)
# ======================================================================


class RFModulationOperator(nn.Module):
    r"""Block-diagonal RF modulation operator for multi-coil MRI.

    Models the combined effect of transmit RF inhomogeneity (:math:`B_1^+`)
    and receive coil sensitivity (:math:`B_1^-`) on the acquired signal:

    .. math::

        \mathcal{C}(\mathbf{x}) = \text{diag}(B_1^+ \cdot B_1^-) \cdot \mathbf{x}

    For multi-coil data, each coil sees the image modulated by its
    individual sensitivity profile:

    .. math::

        y_c(\mathbf{r}) = S_c(\mathbf{r}) \cdot x(\mathbf{r})

    The adjoint (inverse) normalises intensity variations.

    Args:
        num_coils: Number of receive coils (1 for single-coil).
    """

    def __init__(self, num_coils: int = 1) -> None:
        super().__init__()
        self.num_coils = num_coils

    def forward(
        self,
        image: torch.Tensor,
        b1_plus: torch.Tensor,
        b1_minus: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RF modulation (forward model).

        Args:
            image: Image ``[B, C, H, W]`` (real or complex).
            b1_plus: Transmit field ``[B, 1, H, W]`` (scale map, 1.0 = ideal).
            b1_minus: Receive sensitivity ``[B, num_coils, H, W]`` or
                      ``[B, 1, H, W]`` for single-coil.

        Returns:
            Modulated image ``[B, num_coils, H, W]``.

        forward method for RFModulationOperator.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b1_plus (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            b1_minus (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Transmit modulation: image * B1+
        modulated = image * b1_plus

        # Receive modulation: per-coil sensitivity
        if self.num_coils > 1 and b1_minus.shape[1] > 1:
            # Expand image channels to match coils
            if modulated.shape[1] == 1:
                modulated = modulated.expand(-1, self.num_coils, -1, -1)
            return modulated * b1_minus
        return modulated * b1_minus

    def adjoint(
        self,
        coil_images: torch.Tensor,
        b1_plus: torch.Tensor,
        b1_minus: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RF modulation adjoint (coil combination + B1 normalisation).

        Args:
            coil_images: Multi-coil images ``[B, num_coils, H, W]``.
            b1_plus: Transmit field ``[B, 1, H, W]``.
            b1_minus: Receive sensitivity ``[B, num_coils, H, W]``.

        Returns:
            Combined image ``[B, 1, H, W]``.
        """
        # SENSE-like combination: x = Σ(S_c* · y_c) / Σ(|S_c|²)
        numerator = (coil_images * b1_minus.conj()).sum(dim=1, keepdim=True)
        denominator = (b1_minus.abs().pow(2)).sum(dim=1, keepdim=True).clamp(min=1e-6)
        combined = numerator / denominator

        # Compensate B1+ transmit
        return combined / b1_plus.clamp(min=1e-4)


# ======================================================================
# 2. NUFFT Trajectory Auto-Calibration Operator  (G)
# ======================================================================


class NUFFTTrajectoryCalibrator(nn.Module):
    r"""Calibrate gradient trajectory delays from marker sharpness.

    Gradient hardware delays cause the actual k-space trajectory to
    deviate from the intended one.  Using the known marker cross
    geometry as a sharpness metric, we optimise delay parameters
    until the reconstructed marker matches the ideal prior:

    .. math::

        \Delta\tau^* = \arg\min_{\Delta\tau}
            \| \text{NUFFT}^{-1}(y; k(\Delta\tau)) \cdot M_{\text{marker}}
               - P_{\text{cross}} \|_2^2

    For Cartesian acquisitions, delays manifest as linear phase ramps
    per readout line.  This module estimates and corrects those delays.

    Args:
        im_size: Image dimensions ``(H, W)``.
        max_delay_us: Maximum gradient delay to search (microseconds).
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        max_delay_us: float = 5.0,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.max_delay_us = max_delay_us

        # Learnable delay parameters: [readout_delay, phase_delay]
        self.delays = nn.Parameter(torch.zeros(2))

    def compute_phase_correction(
        self,
        delays: torch.Tensor,
    ) -> torch.Tensor:
        """Build phase correction matrix from delay parameters.

        .. note::

            ``delays`` are applied as ``phase = 2π·k·delay`` on the normalized
            k-grid, i.e. each entry is a **dimensionless k-space shift** (≈ pixel
            shift), NOT a physical microsecond delay — there is no dwell-time /
            bandwidth conversion. The ``_us`` naming and the ``max_delay_us``
            bound are therefore in normalized-shift units; wire a
            ``dwell_time_s`` conversion if a true µs interpretation is needed.

        Args:
            delays: ``[2]`` tensor (readout shift, phase-encode shift), in
                normalized k-grid units.

        Returns:
            Phase correction ``[1, 1, H, W]`` complex.
        """
        H, W = self.im_size

        # Delays convert to linear phase ramps in k-space. Use the centered
        # fftshift(fftfreq) grid (step 1/N, exact DC zero), NOT linspace(-0.5,
        # 0.5, N) (step 1/(N-1), no exact DC) — see RigidKinematicOperator.
        kx = torch.fft.fftshift(torch.fft.fftfreq(W, device=delays.device))
        ky = torch.fft.fftshift(torch.fft.fftfreq(H, device=delays.device))

        # Phase ramp from the (normalized) k-shift; delays[0] is a dimensionless
        # shift, not microseconds (see the docstring note).
        phase_x = 2.0 * math.pi * kx * delays[0]
        phase_y = 2.0 * math.pi * ky * delays[1]

        # 2D correction
        correction = torch.exp(1j * (phase_x.unsqueeze(0) + phase_y.unsqueeze(1)))  # [H, W]

        return correction.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def calibrate(
        self,
        kspace: torch.Tensor,
        marker_mask: torch.Tensor,
        marker_prior: torch.Tensor,
        num_iters: int = 50,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """Optimise delay parameters using marker sharpness as objective.

        Args:
            kspace: Acquired k-space ``[B, 1, H, W]`` complex.
            marker_mask: Marker region mask ``[1, 1, H, W]``.
            marker_prior: Ideal marker image ``[1, 1, H, W]``.
            num_iters: Optimisation iterations.
            lr: Learning rate for gradient descent.

        Returns:
            Optimised delays ``[2]``.
        """
        delays = self.delays.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([delays], lr=lr)

        for _it in range(num_iters):
            optimizer.zero_grad()
            correction = self.compute_phase_correction(delays)
            corrected_kspace = kspace * correction
            image = torch.fft.ifft2(corrected_kspace, dim=(-2, -1))
            image_mag = image.abs()

            # Sharpness loss: difference from ideal marker
            loss = F.mse_loss(
                image_mag * marker_mask,
                marker_prior.expand_as(image_mag) * marker_mask,
            )
            loss.backward()
            optimizer.step()

            # Clamp delays to physical range
            with torch.no_grad():
                delays.clamp_(-self.max_delay_us, self.max_delay_us)

        with torch.no_grad():
            self.delays.copy_(delays)

        return delays.detach()

    def forward(
        self,
        kspace: torch.Tensor,
    ) -> torch.Tensor:
        """Apply calibrated trajectory correction to k-space.

        Args:
            kspace: Complex k-space ``[B, 1, H, W]``.

        Returns:
            Corrected k-space ``[B, 1, H, W]``.

        forward method for NUFFTTrajectoryCalibrator.

        Executes PyTorch tensor operations.

        Args:
            kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        correction = self.compute_phase_correction(self.delays)
        return kspace * correction


__all__ = [
    "NUFFTTrajectoryCalibrator",
    "RFModulationOperator",
]
