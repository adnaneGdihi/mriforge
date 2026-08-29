import torch
import torch.nn as nn

from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c


class PhotonicFFT2d(nn.Module):
    """
    Paradigm 6: Photonic Hardware Simulator.
    Simulates FFT with optical noise (phase jitter, thermal noise).
    Can be used as a drop-in 'Physical Layer' in MRI reconstruction networks.
    """

    def __init__(self, phase_noise_std=0.01, thermal_noise_std=0.005, bit_depth=8):
        """__init__.

        Args:
            phase_noise_std (Any): Description.
            thermal_noise_std (Any): Description.
            bit_depth (Any): Description.
        """
        super().__init__()
        self.phase_noise_std = phase_noise_std
        self.thermal_noise_std = thermal_noise_std
        self.bit_depth = bit_depth

    def quantize(self, tensor):
        # Simulate DAC/ADC bit depth limitations
        """quantize.

        Args:
            tensor (Any): Description.
        Returns:
            Any: Description.
        """
        if self.bit_depth is None:
            return tensor
        min_val, max_val = tensor.min(), tensor.max()
        scale = (2**self.bit_depth - 1) / (max_val - min_val + 1e-8)
        return torch.round((tensor - min_val) * scale) / scale + min_val

    def forward(self, x):
        # x: [Batch, C, H, W] (Real/Image or Magnitude)
        # 1. DAC (Quantization)
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PhotonicFFT2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.quantize(x)

        # 2. Optical FFT (Analog Fourier Transform)
        # Input assumed real, output complex k-space
        if x.is_complex():
            kspace = fft2c(x)
        else:
            kspace = fft2c(x.to(torch.complex64))

        # 3. Optical Noise (Phase + Thermal)
        if self.training:
            # Phase Jitter (Optical path length variations)
            noise_phase = torch.randn_like(kspace.real) * self.phase_noise_std
            phase_rot = torch.exp(1j * noise_phase)

            # Thermal Noise (Detector noise)
            thermal = torch.randn_like(kspace) * self.thermal_noise_std

            # Apply degradations
            kspace = (kspace * phase_rot) + thermal

        # 4. Optical IFFT + ADC (Readout)
        # Reconstruct back to image domain simulate end-to-end optical processor
        recon = ifft2c(kspace)
        return self.quantize(torch.abs(recon))
