#!/usr/bin/env python3
"""VQ-VAE Generator implementation for MRI super-resolution.

This module implements a VQ-VAE generator that follows the
IGenerator interface, providing discrete latent representations
with improved perceptual quality.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="vqvae_generator", training_mode="vae")
class VQVAEGenerator(nn.Module, IGenerator):
    """VQ-VAE Generator implementing IGenerator interface.

    Uses vector quantization with discrete codebook for improved
    perceptual quality in MRI super-resolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
    ):
        """Initialize VQ-VAE Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            num_embeddings: Number of codebook entries
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate for codebook updates
            hidden_dims: Hidden dimensions for encoder/decoder

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.hidden_dims = hidden_dims

        # Build encoder
        self.encoder = self._build_encoder()

        # Build decoder
        self.decoder = self._build_decoder()

        # Vector quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

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
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                ],
            )
            current_channels = h_dim

        # Final projection to latent dimension
        layers.append(nn.Conv2d(current_channels, self.latent_dim, kernel_size=1))

        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network."""
        layers = []

        # Initial projection
        current_channels = self.latent_dim
        layers.append(nn.Conv2d(current_channels, self.hidden_dims[-1], kernel_size=1))
        layers.append(nn.ReLU(inplace=True))
        current_channels = self.hidden_dims[-1]  # Update current_channels

        # Upsampling layers
        reversed_dims = list(reversed(self.hidden_dims))
        for _i, h_dim in enumerate(reversed_dims):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels,
                        h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                ],
            )
            current_channels = h_dim

        # Final output layer
        layers.append(nn.Conv2d(current_channels, self.out_channels, kernel_size=1))
        layers.append(nn.Tanh())  # Output in [-1, 1] range

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Weight initialization handled by centralized weight_initialization module."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VQ-VAE.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Reconstructed output [B, C, H, W]

        forward method for VQVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, vq_loss, perplexity = self.quantizer(z)

        # Decode
        x_hat = self.decoder(z_q)

        # Store losses for training
        self.vq_loss = vq_loss
        self.perplexity = perplexity

        return x_hat

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate from latent space.

        Args:
            z: Latent vectors [B, latent_dim, H, W] or [B, latent_dim]
            **kwargs: Additional generation parameters

        Returns:
            Generated images [B, out_channels, H, W]

        """
        # If z is flattened, reshape to spatial dimensions
        if z.dim() == 2:
            # Assume square spatial dimension based on encoder output
            total_elements = z.shape[0] * z.shape[1]
            spatial_elements = total_elements // (self.latent_dim * z.shape[0])
            spatial_size = int(math.sqrt(spatial_elements))
            z = z.view(z.shape[0], self.latent_dim, spatial_size, spatial_size)

        # Quantize and decode
        z_q, _, _ = self.quantizer(z)
        return self.decoder(z_q)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to output space."""
        return self.decoder(z)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "VQVAEGenerator"


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA updates."""

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """__init__.

        Args:
            num_embeddings (int): Description.
            embedding_dim (int): Description.
            commitment_cost (float): Description.
            decay (float): Description.
            epsilon (float): Description.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embeddings.clone())

        self.reset_parameters()

    def reset_parameters(self):
        """Reset codebook parameters."""
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through quantizer.

        Args:
            z: Input tensor [B, C, H, W]

        Returns:
            Tuple of (quantized, loss, indices)

        forward method for VectorQuantizer.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Flatten
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Compute distances
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.t())
        )

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=z.device,
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(z.shape)

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # Losses
        codebook_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized_st, z.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # EMA update
        if self.training:
            self.ema_update(encodings, z_flattened)

        return quantized_st, loss, encoding_indices

    def ema_update(self, encodings: torch.Tensor, z_flattened: torch.Tensor):
        """EMA update for codebook."""
        encodings_sum = torch.sum(encodings, dim=0)
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * encodings_sum
        self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * torch.matmul(
            encodings.t(), z_flattened
        )

        n = torch.sum(self.ema_count)
        self.ema_count = (
            (self.ema_count + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        )
        self.embeddings = self.ema_weight / self.ema_count.unsqueeze(1)


def create_vqvae_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 64,
    num_embeddings: int = 512,
    commitment_cost: float = 0.25,
    decay: float = 0.99,
    hidden_dims: tuple[int, ...] = (64, 128, 256),
    **kwargs,
) -> VQVAEGenerator:
    """Factory function for creating VQ-VAE generators.

    Args:
        in_channels: Number of input/output channels
        out_channels: Number of output classes (unused for VQ-VAE)
        latent_dim: Dimension of latent space
        num_embeddings: Number of codebook entries
        commitment_cost: Weight for commitment loss
        decay: EMA decay rate for codebook updates
        hidden_dims: Hidden dimensions for encoder/decoder
        **kwargs: Additional arguments passed to VQVAEGenerator

    Returns:
        Configured VQ-VAE generator

    """
    return VQVAEGenerator(
        in_channels=in_channels,
        out_channels=in_channels,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay,
        hidden_dims=hidden_dims,
    )
