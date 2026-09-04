"""Vector Quantized Generative Adversarial Network (VQGAN)

This module implements VQGAN for high-quality image generation using
discrete latent representations with adversarial training.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from .vqvae import VQVAE


@register_model(name="vqgan", training_mode="gan")
class VQGAN(nn.Module, IGenerator):
    """Vector Quantized Generative Adversarial Network for MRI super-resolution.

    VQGAN combines VQ-VAE with adversarial training to produce high-quality
    reconstructions. The discriminator helps improve perceptual quality
    while the VQ-VAE ensures discrete latent representations.

    Architecture:
    - Generator: VQ-VAE model
    - Discriminator: Patch-based discriminator for adversarial training

    Reference:
    Esser et al. "Taming Transformers for High-Resolution Image Synthesis"
    CVPR 2021
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hiddein_channels: int = 128,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 1.0,
        num_layers: int = 4,
        downsample_factor: int = 8,
        disc_channels: int = 64,
        disc_layers: int = 4,
        use_spectral_norm: bool = True,
    ):
        """Initialize VQGAN.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hiddein_channels: Number of hidden channels in VQ-VAE
            num_embeddings: Number of codebook entries
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
            num_layers: Number of VQ-VAE encoder/decoder layers
            downsample_factor: Total downsampling factor for VQ-VAE
            disc_channels: Number of channels in discriminator
            disc_layers: Number of discriminator layers
            use_spectral_norm: Whether to use spectral normalization

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Generator (VQ-VAE)
        self.generator = VQVAE(
            in_channels=in_channels,
            out_channels=out_channels,
            hiddein_channels=hiddein_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            num_layers=num_layers,
            downsample_factor=downsample_factor,
        )

        # Discriminator
        self.discriminator = self._build_discriminator(
            in_channels=out_channels,
            channels=disc_channels,
            num_layers=disc_layers,
            use_spectral_norm=use_spectral_norm,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _build_discriminator(
        self,
        in_channels: int,
        channels: int,
        num_layers: int,
        use_spectral_norm: bool,
    ) -> nn.Sequential:
        """Build patch-based discriminator."""
        layers = []

        # Initial convolution
        conv_layer = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        if use_spectral_norm:
            conv_layer = nn.utils.spectral_norm(conv_layer)
        layers.append(conv_layer)
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Downsampling layers
        current_channels = channels
        for _i in range(num_layers - 1):
            next_channels = min(current_channels * 2, channels * 8)
            conv_layer = nn.Conv2d(
                current_channels,
                next_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if use_spectral_norm:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = next_channels

        # Final convolution
        conv_layer = nn.Conv2d(current_channels, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            conv_layer = nn.utils.spectral_norm(conv_layer)
        layers.append(conv_layer)

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "vqgan"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VQGAN generator.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Reconstructed tensor of shape (B, out_channels, H, W)

        """
        return self.generator(x)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space.

        Args:
            z: Latent tensor of shape (B, embedding_dim, H', W')
            **kwargs: Additional generation parameters

        Returns:
            Generated tensor of shape (B, out_channels, H, W)

        """
        return self.generator.generate(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape.

        Args:
            input_shape: Input tensor shape (B, C, H, W)

        Returns:
            Output tensor shape (B, out_channels, H, W)

        """
        return self.generator.get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Discriminator output of shape (B, 1, H', W')

        """
        return self.discriminator(x)

    def get_generator(self) -> VQVAE:
        """Get the VQ-VAE generator."""
        return self.generator

    def get_discriminator(self) -> nn.Sequential:
        """Get the discriminator network."""
        return self.discriminator

    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook usage statistics."""
        return self.generator.get_codebook_usage()

    def get_codebook_entropy(self) -> float:
        """Get codebook entropy."""
        return self.generator.get_codebook_entropy()

    def sample_from_codebook(
        self,
        num_samples: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample random vectors from the codebook.

        Args:
            num_samples: Number of samples to generate
            device: Device to place samples on

        Returns:
            Sampled latent vectors

        """
        return self.generator.sample_from_codebook(num_samples, device)

    def reconstruct_from_indices(
        self,
        indices: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct image from codebook indices.

        Args:
            indices: Codebook indices of shape (B, H', W')
            spatial_size: Spatial dimensions (H, W) for upsampling

        Returns:
            Reconstructed images

        """
        return self.generator.reconstruct_from_indices(indices, spatial_size)
