"""Noise Simulation for MRI Physics.

Implements realistic noise models:
- Gaussian (White Noise): standard thermal noise in complex k-space.
- Rician: Magnitude of complex image with Gaussian noise (low SNR bias).
- Chi-Square: Multi-coil SoS combination (non-central Chi).

References:
    Gudbjartsson, H., & Patz, S. (1995). The Rician distribution of noisy MRI data. MRM.
"""

import math
from typing import Literal

import torch
import torch.nn as nn


class NoiseSimulator(nn.Module):
    r"""MRI Noise Simulator.

    Simulates noise in:
    1. K-space (Complex Gaussian)
    2. Image magnitude (Rician)

    Args:
        snr_range: Tuple of (min, max) SNR in dB
        noise_type: 'gaussian', 'rician'
    Mathematical Formulation:
    .. math::

        \mathcal{O}_{Noise}(x) = x + \mathcal{N}(0, \sigma^2)"""

    def __init__(
        self,
        snr_range: tuple[float, float] = (10.0, 30.0),
        noise_type: Literal["gaussian", "rician"] = "rician",
    ):
        """__init__.

        Args:
            snr_range (tuple[float, float]): Description.
            noise_type (Literal['gaussian', 'rician']): Description.
        """
        super().__init__()
        self.snr_range = snr_range
        self.noise_type = noise_type

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Add noise to signal.

        Args:
            signal: Input tensor (B, C, H, W) - usually complex k-space or image

        Returns:
            Noisy tensor
        """
        batch_size = signal.shape[0]
        device = signal.device

        # Sample SNR for batch
        min_snr, max_snr = self.snr_range
        snr_db = torch.rand(batch_size, 1, 1, 1, device=device) * (max_snr - min_snr) + min_snr
        snr_linear = 10 ** (snr_db / 20.0)  # Amplitude ratio

        # Calculate signal power (per sample magnitude)
        # Robust estimation: use 95th percentile or mean of non-zero
        sig_mag = torch.abs(signal)
        signal_level = torch.quantile(sig_mag.flatten(1), 0.95, dim=1).view(batch_size, 1, 1, 1)

        # Noise sigma
        sigma = signal_level / (snr_linear + 1e-8)

        # Generate Complex Gaussian Noise
        noise = torch.randn_like(signal) + 1j * torch.randn_like(signal)
        # Adjust scaling: for complex Gaussian, var = 2*sigma^2
        # We want standard deviation of real/imag component to be sigma/sqrt(2) ?
        # Convention: sigma is std of Gaussian on real/imag axes separately
        noise = noise * (sigma / math.sqrt(2))

        if self.noise_type == "gaussian":
            # Additive Complex Gaussian (K-space or Complex Image)
            return signal + noise

        elif self.noise_type == "rician":
            # Rician: | S + N |
            # Only valid for Magnitude Images
            # If input is complex, we add noise then take magnitude
            noisy_complex = signal + noise
            return torch.abs(noisy_complex)

        return signal

    def add_rician_noise(self, magnitude_image: torch.Tensor, snr: float) -> torch.Tensor:
        """Add Rician noise to a magnitude image."""
        sigma = torch.quantile(magnitude_image, 0.95) / (10 ** (snr / 20.0))
        # Split sigma across the two quadratures (sigma/√2 each) so the total
        # complex noise power is sigma² — the amplitude-SNR convention used by
        # ``forward``. Putting sigma on each quadrature (the pre-fix code) gives
        # total power 2·sigma², i.e. √2 (≈3 dB) too much noise and disagrees
        # with ``forward`` in this same class.
        scale = sigma / math.sqrt(2)
        noise = scale * torch.randn_like(magnitude_image) + 1j * scale * torch.randn_like(
            magnitude_image
        )
        return torch.abs(magnitude_image.to(torch.complex64) + noise)
