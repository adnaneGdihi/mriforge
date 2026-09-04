"""FourierKAN Decoder for VAE
===========================

Decoder using Fourier-based KAN layers for frequency domain
feature reconstruction in VAEs.
"""

import torch
from torch import nn

from ..layers.kan.fourier_basis import FourierKANConv2DLayer


class FourierKANDecoder(nn.Module):
    """VAE Decoder with FourierKAN layers."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 256,
        num_bases: int = 8,
        max_frequency: float = 1.0,
        basis_trainable: bool = True,
        output_size: int = 64,  # Target output spatial size
        use_complex_conv: bool = False,
        use_gabor: bool = False,
    ):
        """__init__.

        Args:
            out_channels (int): Description.
            latent_dim (int): Description.
            num_bases (int): Description.
            max_frequency (float): Description.
            basis_trainable (bool): Description.
            output_size (int): Description.
            use_complex_conv (bool): Description.
            use_gabor (bool): Description.
        """
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.output_size = output_size

        # Calculate intermediate sizes based on output_size
        # We use 3 upsampling layers, so start_size = output_size // 8
        start_size = max(4, output_size // 8)  # Minimum 4x4
        self.start_size = start_size

        # Latent to feature space
        self.fc_expand = nn.Linear(latent_dim, 512 * start_size * start_size)

        # Decoder layers
        self.dec1 = FourierKANConv2DLayer(
            512,
            256,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.dec2 = FourierKANConv2DLayer(
            256,
            128,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.dec3 = FourierKANConv2DLayer(
            128,
            64,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.dec4 = FourierKANConv2DLayer(
            64,
            out_channels,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Expand latent to feature space
        """forward.

        Args:
            z (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FourierKANDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.fc_expand(z)
        x = x.view(-1, 512, self.start_size, self.start_size)

        # Decoder with dynamic upsampling
        x = self.upsample(self.dec1(x))  # start_size -> 2*start_size
        x = self.upsample(self.dec2(x))  # 2*start_size -> 4*start_size
        x = self.upsample(self.dec3(x))  # 4*start_size -> 8*start_size

        # Final convolution (no upsampling to reach target size)
        x = self.dec4(x)  # 8*start_size -> 8*start_size

        # If we haven't reached the target size, do one more upsampling
        if 8 * self.start_size != self.output_size:
            x = self.upsample(x)

        return x
