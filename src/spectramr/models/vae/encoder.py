#!/usr/bin/env python3
"""VAE Encoder implementation for MRI super-resolution.

This module implements a variational encoder that maps input images
to a latent distribution parameterized by mean and log-variance.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IModel


class VAEEncoder(nn.Module, IModel):
    """Variational Autoencoder Encoder.

    Maps input images to a latent distribution (mean, log-variance)
    using a convolutional encoder network.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """Initialize VAE Encoder.

        Args:
            in_channels: Number of input channels
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for encoder layers
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization

        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build encoder network
        self.encoder = self._build_encoder()

        # Use adaptive pooling to ensure fixed flatten size regardless of input resolution
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fixed flattened size: last hidden dim * 4 * 4
        last_hidden_dim = self.hidden_dims[-1] if self.hidden_dims else self.in_channels
        flattened_size = last_hidden_dim * 4 * 4

        # Final projection to latent parameters
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_var = nn.Linear(flattened_size, latent_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network."""
        layers = []
        current_channels = self.in_channels

        for h_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(
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

        return nn.Sequential(*layers)

    def _get_encoder_layers_config(self) -> list:
        """Get encoder layers configuration for shape inference."""
        layers = []
        current_channels = self.in_channels

        for h_dim in self.hidden_dims:
            layers.append(
                {
                    "type": "conv2d",
                    "in_channels": current_channels,
                    "out_channels": h_dim,
                    "kernel_size": 4,
                    "stride": 2,
                    "padding": 1,
                },
            )
            current_channels = h_dim

        return layers

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (mu, log_var) for latent distribution

        forward method for VAEEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        x = self.encoder(x)

        # Adaptive pooling to fix spatial dimension to 4x4
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.forward(x)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]

        Returns:
            Sampled latent vector [B, latent_dim]

        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Sample latent vector from input."""
        mu, log_var = self.encode(x)
        return self.reparameterize(mu, log_var)

    def get_latent_distribution(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get latent distribution parameters from input."""
        return self.encode(x)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "VAEEncoder"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the encoder."""
        return sum(p.numel() for p in self.parameters())


def create_vae_encoder(
    in_channels: int = 1,
    latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
) -> VAEEncoder:
    """Factory function for creating VAE encoders.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder layers

    Returns:
        Configured VAE encoder

    """
    return VAEEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
