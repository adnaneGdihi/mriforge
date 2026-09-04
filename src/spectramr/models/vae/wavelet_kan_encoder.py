"""WaveletKAN Encoder for VAE
===========================

Encoder using Wavelet-based KAN layers for multi-scale
feature extraction in VAEs.
"""

import torch
from torch import nn

from spectramr.models.constants import (
    DEFAULT_CHANNEL_PROGRESSION,
    DEFAULT_KAN_NUM_BASES,
    DEFAULT_POOL_STRIDE,
    DEFAULT_VAE_LATENT_DIM,
    DEFAULT_WAVELET_FAMILY,
)
from spectramr.models.shape_inference import (
    WAVELET_KAN_ENCODER_LAYERS,
    compute_flattened_size,
    infer_conv_stack_output,
)

from ..layers.kan.wavelet_basis import WaveletKANConv2DLayer


class WaveletKANEncoder(nn.Module):
    """VAE Encoder with WaveletKAN layers."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = DEFAULT_VAE_LATENT_DIM,
        num_bases: int = DEFAULT_KAN_NUM_BASES,
        wavelet_family: str = DEFAULT_WAVELET_FAMILY,
        basis_trainable: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            num_bases (int): Description.
            wavelet_family (str): Description.
            basis_trainable (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder layers with channel progression from constants
        channels = DEFAULT_CHANNEL_PROGRESSION
        self.enc1 = WaveletKANConv2DLayer(
            in_channels,
            channels[0],
            num_bases=num_bases,
            wavelet_family=wavelet_family,
            basis_trainable=basis_trainable,
        )
        self.enc2 = WaveletKANConv2DLayer(
            channels[0],
            channels[1],
            num_bases=num_bases,
            wavelet_family=wavelet_family,
            basis_trainable=basis_trainable,
        )
        self.enc3 = WaveletKANConv2DLayer(
            channels[1],
            channels[2],
            num_bases=num_bases,
            wavelet_family=wavelet_family,
            basis_trainable=basis_trainable,
        )
        self.enc4 = WaveletKANConv2DLayer(
            channels[2],
            channels[3],
            num_bases=num_bases,
            wavelet_family=wavelet_family,
            basis_trainable=basis_trainable,
        )

        # Pooling with stride from constants
        self.pool = nn.MaxPool2d(DEFAULT_POOL_STRIDE)

        # Calculate flattened size analytically
        h_out, w_out, c_out = infer_conv_stack_output(
            h=64,
            w=64,
            layers=WAVELET_KAN_ENCODER_LAYERS,
        )
        flattened_size = compute_flattened_size(h_out, w_out, c_out)

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

        forward method for WaveletKANEncoder.

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

        # Flatten
        x = self.flatten(x)

        # If the flattened size doesn't match expected, recreate linear layers
        if x.shape[1] != self.fc_mu.in_features:
            self.fc_mu = nn.Linear(x.shape[1], self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(x.shape[1], self.latent_dim).to(x.device)

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
