"""FourierKAN Encoder for VAE
===========================

Encoder using Fourier-based KAN layers for frequency domain
feature extraction in VAEs.
"""

import torch
from torch import nn

from spectramr.models.constants import (
    DEFAULT_CHANNEL_PROGRESSION,
    DEFAULT_FOURIER_MAX_FREQUENCY,
    DEFAULT_KAN_NUM_BASES,
    DEFAULT_POOL_STRIDE,
    DEFAULT_VAE_LATENT_DIM,
)

from ..layers.kan.fourier_basis import FourierKANConv2DLayer


class FourierKANEncoder(nn.Module):
    """VAE Encoder with FourierKAN layers."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = DEFAULT_VAE_LATENT_DIM,
        num_bases: int = DEFAULT_KAN_NUM_BASES,
        max_frequency: float = DEFAULT_FOURIER_MAX_FREQUENCY,
        basis_trainable: bool = True,
        use_complex_conv: bool = False,
        use_gabor: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            num_bases (int): Description.
            max_frequency (float): Description.
            basis_trainable (bool): Description.
            use_complex_conv (bool): Description.
            use_gabor (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder layers with channel progression from constants
        channels = DEFAULT_CHANNEL_PROGRESSION
        self.enc1 = FourierKANConv2DLayer(
            in_channels,
            channels[0],
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.enc2 = FourierKANConv2DLayer(
            channels[0],
            channels[1],
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.enc3 = FourierKANConv2DLayer(
            channels[1],
            channels[2],
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.enc4 = FourierKANConv2DLayer(
            channels[2],
            channels[3],
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )

        # Pooling with stride from constants
        self.pool = nn.MaxPool2d(DEFAULT_POOL_STRIDE)

        # Use adaptive pooling to ensure fixed flatten size regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fixed flattened size: channels[3] * 4 * 4
        flattened_size = DEFAULT_CHANNEL_PROGRESSION[3] * 4 * 4

        # Flatten and latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for FourierKANEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.enc4(x)

        # Adaptive pool to fixed spatial size
        x = self.adaptive_pool(x)

        # Flatten
        x = self.flatten(x)

        # Latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """reparameterize.

        Args:
            mu (torch.Tensor): Description.
            logvar (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
