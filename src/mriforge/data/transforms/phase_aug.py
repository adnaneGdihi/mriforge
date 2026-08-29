"""Phase augmentation module for MRI data.

Synthesises a smooth, random low-frequency phase map for magnitude-only data so
that complex-valued models see plausible phase structure during training.

Scope / honesty note: this is a phase *augmentation*, NOT a calibrated B0
field-map simulator. The phase is Gaussian-smoothed random noise rescaled so
its std ≈ π; it has no physical Hz calibration and no dependence on field
strength, TE, or a susceptibility source. For physically-grounded off-resonance
(``Δpix = γ·ΔB0·T_readout`` with a real Hz field) use
``data.transforms.realistic_degradations.B0GeometricDistortion`` /
``infrastructure.physics.field_simulation`` instead. Do not treat the phase this
transform produces as a B0 or susceptibility map.
"""

import torch
import torch.nn.functional as F
from torch import nn


class PhaseSimulator(nn.Module):
    """Generate a smooth random low-frequency phase map (augmentation).

    NOT a B0/susceptibility field simulator — see the module docstring. The
    output phase is uncalibrated (std ≈ π) and carries no Hz/field-strength/TE
    dependence; it exists to give complex models plausible phase texture.
    """

    def __init__(self, kernel_size: int = 21, sigma_range: tuple[float, float] = (1.0, 3.0)):
        """__init__.

        Args:
            kernel_size (int): Description.
            sigma_range (tuple[float, float]): Description.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def forward(self, magnitude_img: torch.Tensor) -> torch.Tensor:
        """Apply synthetic phase to magnitude image.

        Args:
            magnitude_img: Real-valued magnitude image [B, C, H, W]

        Returns:
            Complex-valued image [B, C, H, W]

        forward method for PhaseSimulator.

        Executes PyTorch tensor operations.

        Args:
            magnitude_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(magnitude_img):
            return magnitude_img

        B, C, H, W = magnitude_img.shape
        device = magnitude_img.device

        # Generate random low-frequency phase
        # We generate a smaller map and upsample to ensure smoothness (low frequency)
        # Alternatively, use Gaussian blur on full resolution noise

        # Method 1: Gaussian Blur on random noise
        raw_phase = torch.randn(B, C, H, W, device=device)

        # Create Gaussian kernel
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()
        k_size = self.kernel_size
        if k_size % 2 == 0:
            k_size += 1

        # Gaussian kernel generation
        x = torch.arange(k_size, device=device) - k_size // 2
        grid = torch.stack(torch.meshgrid(x, x, indexing="ij"))
        kernel = torch.exp(-(grid**2).sum(0) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k_size, k_size).repeat(C, 1, 1, 1)

        # Apply smoothing
        # Padding to avoid edge artifacts
        pad = k_size // 2
        smooth_phase = F.conv2d(raw_phase, kernel, padding=pad, groups=C)

        # Rescale to std ≈ π so the phase is significant but smooth. This is an
        # arbitrary augmentation scale, NOT a physical B0 amplitude (no Hz/TE
        # calibration) — see the module docstring.
        smooth_phase = smooth_phase / (smooth_phase.std() + 1e-8) * 3.14159

        # Create complex image: M * exp(j * phi)
        phase_map = torch.exp(1j * smooth_phase)
        return magnitude_img.to(dtype=torch.complex64) * phase_map


def apply_synthetic_phase(magnitude_img: torch.Tensor) -> torch.Tensor:
    """Functional interface for phase simulation."""
    simulator = PhaseSimulator()
    return simulator(magnitude_img)
