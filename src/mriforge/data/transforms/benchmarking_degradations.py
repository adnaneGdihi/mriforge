"""
Benchmarking Degradation Vectors for MRI Quality Metrics
========================================================

Mathematically rigorous degradation vectors ($\vec{v}$) designed specifically
for evaluating the robustness and sensitivity of MRI Image Quality Metrics.
These vectors operate primarily in k-space to ensure physically accurate
artifact generation (like Rician noise in the magnitude domain).
"""

import math

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


class KSpaceComplexGaussianNoise(nn.Module):
    """
    Simulates thermal system noise acquired in the receiver coils ($\vec{v}_{noise}$).

    Injects complex Gaussian white noise directly into k-space. When inverse-Fourier
    transformed and magnitude reconstructed, this mathematically correctly forms
    a Rician noise distribution in the image domain.
    """

    def __init__(self, target_snr_db: float = 20.0, seed: int | None = None):
        """
        Args:
            target_snr_db: Target Signal-to-Noise Ratio in Decibels. Lower is noisier.
            seed: Optional seed for reproducibility.
        """
        super().__init__()
        self.target_snr_db = target_snr_db
        self.seed = seed

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply complex noise via k-space.

        Args:
            image: Pristine image tensor [B, C, H, W] (real-valued magnitude or complex).

        Returns:
            Noisy image tensor with the same shape/type as input.

        forward method for KSpaceComplexGaussianNoise.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.target_snr_db == float("inf"):
            return image

        device = image.device

        # Determine if input is already complex or magnitude
        is_magnitude = not torch.is_complex(image)
        complex_img = image + 0j if is_magnitude else image

        # Transform to k-space
        kspace = fft2c(complex_img)

        # Calculate signal power in k-space
        signal_power = torch.mean(torch.abs(kspace) ** 2, dim=(-1, -2), keepdim=True)

        # Calculate noise variance based on desired SNR
        # SNR_linear = 10^(SNR_dB / 10)
        # SNR = Signal_Power / Noise_Power => Noise_Power = Signal_Power / SNR_linear
        snr_linear = 10.0 ** (self.target_snr_db / 10.0)
        noise_power = signal_power / snr_linear
        sigma = torch.sqrt(noise_power / 2.0)  # /2 because it splits across real and imaginary

        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            noise_real = torch.randn(kspace.shape, device=device, generator=generator)
            noise_imag = torch.randn(kspace.shape, device=device, generator=generator)
        else:
            noise_real = torch.randn_like(kspace.real)
            noise_imag = torch.randn_like(kspace.real)

        noise_complex = torch.complex(noise_real, noise_imag) * sigma

        # Add noise in k-space
        kspace_noisy = kspace + noise_complex

        # Transform back to image domain
        img_noisy = ifft2c(kspace_noisy)

        if is_magnitude:
            return torch.abs(img_noisy)
        return img_noisy


class B1TransmitInhomogeneity(nn.Module):
    """
    Simulates $B_1^+$ transmit field bias and global contrast washout ($\vec{v}_{contrast}$).

    Uses a low-frequency 2D Gaussian multiplier map to simulate a non-uniform
    RF excitation profile (e.g., center brightening in 3T+ systems), combined
    with a gamma operation for global contrast shifts.
    """

    def __init__(self, bias_strength: float = 0.5, gamma: float = 1.0):
        """
        Args:
            bias_strength: Intensity of the spatial Gaussian bias. 0 is flat.
            gamma: Contrast shift (gamma < 1 brightens/washes out, gamma > 1 darkens).
        """
        super().__init__()
        self.bias_strength = bias_strength
        self.gamma = gamma

    def _generate_bias_field(self, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        H, W = shape
        # Create coordinates [-1, 1]
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Standard wide 2D Gaussian representing a surface coil or dielectric resonance
        radius_sq = xx**2 + yy**2
        sigma_sq = 0.5  # Wide spread

        gaussian = torch.exp(-radius_sq / (2 * sigma_sq))
        # Scale between [1.0 - bias_strength, 1.0 + bias_strength]
        # At center (r=0), value is 1.0 + bias_strength
        # At edges (r>2), value drops to ~ 1.0 - bias_strength
        # Normalizing gaussian to be roughly [0, 1] then mapping
        bias_field = 1.0 + self.bias_strength * (2.0 * gaussian - 1.0)
        return bias_field.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial bias and gamma correction.

        Args:
            image: Input magnitude image [B, C, H, W] scaled [0, 1].

        Returns:
            Contrast-degraded image.

        forward method for B1TransmitInhomogeneity.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = image.device

        # Gamma correction
        if self.gamma != 1.0:
            # Ensure safe exponentiation for potentially negative tiny numbers due to numerical error
            img_degraded = torch.clamp(image, min=0.0) ** self.gamma
        else:
            img_degraded = image

        # B1 spatial bias
        if self.bias_strength > 0.0:
            bias_field = self._generate_bias_field(image.shape[-2:], device)
            img_degraded = img_degraded * bias_field

        return img_degraded


class KSpaceGhosting(nn.Module):
    """
    Simulates N/2 (Nyquist) ghosting artifacts ($\vec{v}_{artifact}$).

    Induced by applying alternating phase shifts to phase-encode lines in k-space.
    Often caused by EPI readout trajectory imperfections or periodic motion.
    """

    def __init__(self, phase_shift_diff: float = math.pi / 4.0):
        """
        Args:
            phase_shift_diff: The phase difference between even and odd phase-encode
                              lines (in radians). Pi radians causes maximum split.
        """
        super().__init__()
        self.phase_shift_diff = phase_shift_diff

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply alternating phase shifts in k-space.

        Args:
            image: Input image [B, C, H, W]. Assume dimension -2 is phase-encode.

        forward method for KSpaceGhosting.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.phase_shift_diff == 0.0:
            return image

        device = image.device
        is_magnitude = not torch.is_complex(image)
        complex_img = image + 0j if is_magnitude else image

        # FFT to k-space
        kspace = fft2c(complex_img)

        # Create phase shift mask
        H, W = kspace.shape[-2:]
        mask = torch.ones((1, 1, H, 1), dtype=torch.complex64, device=device)

        # Apply phase shift to odd lines (assuming dim -2 H is phase-encode)
        shift_complex = torch.exp(1j * torch.tensor(self.phase_shift_diff, device=device))
        mask[:, :, 1::2, :] = shift_complex

        kspace_shifted = kspace * mask

        # IFFT back
        img_ghosted = ifft2c(kspace_shifted)

        if is_magnitude:
            return torch.abs(img_ghosted)
        return img_ghosted


class KSpaceSpikes(nn.Module):
    """
    Simulates zipper and spike artifacts ($\vec{v}_{artifact}$).

    Injects extreme amplitude delta functions into k-space at random points,
    which manifests as severe striping in the image domain (often caused by
    RF interference or spark-gap discharges).
    """

    def __init__(
        self,
        num_spikes: int = 1,
        spike_intensity_ratio: float = 100.0,
        seed: int | None = None,
    ):
        """
        Args:
            num_spikes: Number of individual k-space spikes to inject.
            spike_intensity_ratio: Multiplier relative to the DC (center) k-space peak.
            seed: Optional random seed.
        """
        super().__init__()
        self.num_spikes = num_spikes
        self.spike_intensity_ratio = spike_intensity_ratio
        self.seed = seed

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inject spikes into k-space.

        Args:
            image: Input image [B, C, H, W].

        forward method for KSpaceSpikes.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.num_spikes == 0 or self.spike_intensity_ratio == 0.0:
            return image

        device = image.device
        is_magnitude = not torch.is_complex(image)
        complex_img = image + 0j if is_magnitude else image

        kspace = fft2c(complex_img)
        B, C, H, W = kspace.shape

        # Calculate maximum magnitude (DC component) to anchor the spike intensity
        dc_amp = torch.max(torch.abs(kspace)).item()
        spike_val = dc_amp * self.spike_intensity_ratio

        # Create copy to mutate
        kspace_spiked = kspace.clone()

        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
        else:
            generator = None

        for b in range(B):
            for c in range(C):
                for _ in range(self.num_spikes):
                    # Pick random frequency point using generator
                    if generator is not None:
                        fy = torch.randint(0, H, (1,), generator=generator, device=device).item()
                        fx = torch.randint(0, W, (1,), generator=generator, device=device).item()
                        phase = (
                            torch.rand(1, generator=generator, device=device).item() * 2 * math.pi
                        )
                    else:
                        fy = torch.randint(0, H, (1,), device=device).item()
                        fx = torch.randint(0, W, (1,), device=device).item()
                        phase = torch.rand(1, device=device).item() * 2 * math.pi

                    # Avoid the exact DC center (H//2, W//2) as a spike there just causes uniform offset
                    if fy == H // 2 and fx == W // 2:
                        fy = (fy + 5) % H

                    kspace_spiked[b, c, fy, fx] += spike_val * torch.exp(
                        1j * torch.tensor(phase, device=device)
                    )

        img_spiked = ifft2c(kspace_spiked)

        if is_magnitude:
            return torch.abs(img_spiked)
        return img_spiked
