#!/usr/bin/env python3
"""VAE Decoder implementation for MRI super-resolution.

This module implements a variational decoder that reconstructs images
from latent vectors using a deconvolutional network.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IModel


class VAEDecoder(nn.Module, IModel):
    """Variational Autoencoder Decoder.

    Reconstructs images from latent vectors using a deconvolutional
    network with skip connections and output in [-1, 1] range.
    """

    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Initialize VAE Decoder.

        Args:
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for decoder layers
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization

        """
        super().__init__()

        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Initial projection from latent to feature map
        self.fc_expand = nn.Linear(latent_dim, hidden_dims[0] * 4)

        # Build decoder network
        self.decoder = self._build_decoder()

        # Initialize weights
        self.apply(self._init_weights)

    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network."""
        layers = []

        # Initial feature map reshaping is handled in forward pass
        current_channels = self.hidden_dims[0]

        # Upsampling layers
        for _i, h_dim in enumerate(self.hidden_dims):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim) if self.use_batch_norm else nn.Identity(),
                    nn.ReLU(inplace=True),
                    (nn.Dropout2d(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()),
                ],
            )
            current_channels = h_dim

        # Final output layer
        layers.append(nn.Conv2d(current_channels, self.out_channels, kernel_size=1))
        layers.append(nn.Tanh())  # Output in [-1, 1] range

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Reconstructed image [B, out_channels, H, W]

        forward method for VAEDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Expand latent to feature map
        x = self.fc_expand(z)
        # Assuming 2x2 spatial from encoder bottleneck
        x = x.view(x.shape[0], self.hidden_dims[0], 2, 2)

        # Decode
        x = self.decoder(x)

        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        return self.forward(z)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from latent vector."""
        return self.forward(z)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "VAEDecoder"


def create_vae_decoder(
    out_channels: int = 1,
    latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
) -> VAEDecoder:
    """Factory function for creating VAE decoders.

    Args:
        out_channels: Number of output channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for decoder layers

    Returns:
        Configured VAE decoder

    """
    return VAEDecoder(
        out_channels=out_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
