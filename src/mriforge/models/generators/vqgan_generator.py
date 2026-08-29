#!/usr/bin/env python3
"""VQGAN Generator implementation for MRI super-resolution.

This module implements a VQGAN generator that follows the
IGenerator interface, combining VQ-VAE with adversarial training
for high-quality reconstruction.

Reference:
Esser et al. "Taming Transformers for High-Resolution Image Synthesis" CVPR 2021
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

from ..vq.vqgan import VQGAN


@register_model(name="vqgan_generator", training_mode="gan")
class VQGANGenerator(nn.Module, IGenerator):
    """VQGAN Generator implementing IGenerator interface.

    Combines VQ-VAE with adversarial training for improved perceptual
    quality in MRI super-resolution. The discriminator helps enforce
    high-frequency details while the VQ codebook ensures discrete
    latent representations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        codebook_size: int = 1024,
        embedding_dim: int = 256,
        hiddein_channels: int = 128,
        num_embeddings: int = 1024,
        commitment_cost: float = 0.25,
        num_layers: int = 4,
        downsample_factor: int = 8,
        disc_channels: int = 64,
        disc_layers: int = 4,
        use_spectral_norm: bool = True,
        ema_decay: float = 0.999,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.35,
        lpips_weight: float = 0.15,
        adversarial_weight: float = 0.05,
        **kwargs,
    ):
        """Initialize VQGAN Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            codebook_size: Alias for num_embeddings (number of codebook entries)
            embedding_dim: Dimension of each codebook entry
            hiddein_channels: Number of hidden channels in encoder/decoder
            num_embeddings: Number of codebook entries (alternative parameter name)
            commitment_cost: Weight for commitment loss
            num_layers: Number of encoder/decoder layers
            downsample_factor: Total downsampling factor
            disc_channels: Number of discriminator channels
            disc_layers: Number of discriminator layers
            use_spectral_norm: Whether to use spectral normalization
            ema_decay: EMA decay rate for codebook updates
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            lpips_weight: Weight for LPIPS loss
            adversarial_weight: Weight for adversarial loss
            **kwargs: Additional keyword arguments (ignored for compatibility)

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Handle parameter aliasing and defaults
        actual_num_embeddings = num_embeddings
        if codebook_size != 1024 or num_embeddings == 1024:
            # If codebook_size is explicitly set (not default), use it
            if codebook_size != 1024:
                actual_num_embeddings = codebook_size

        actual_embedding_dim = embedding_dim
        if embedding_dim != 256 or len(kwargs.get("embedding_dim", [])) == 0:
            actual_embedding_dim = embedding_dim

        # Store loss weights
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.lpips_weight = lpips_weight
        self.adversarial_weight = adversarial_weight
        self.ema_decay = ema_decay

        # Initialize underlying VQGAN
        self.model = VQGAN(
            in_channels=in_channels,
            out_channels=out_channels,
            hiddein_channels=hiddein_channels,
            num_embeddings=actual_num_embeddings,
            embedding_dim=actual_embedding_dim,
            commitment_cost=commitment_cost,
            num_layers=num_layers,
            downsample_factor=downsample_factor,
            disc_channels=disc_channels,
            disc_layers=disc_layers,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, x, **kwargs):
        """Forward pass through VQGAN.

        Args:
            x: Input tensor of shape (B, C, H, W)
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Reconstructed output tensor of same shape as input

        """
        return self.model(x, **kwargs)

    def generate(self, z: "torch.Tensor", **kwargs) -> "torch.Tensor":
        """Generate from latent space.

        Args:
            z: Latent vectors or noise tensor
            **kwargs: Additional generation parameters

        Returns:
            Generated images

        """
        # If z is provided, decode it; otherwise use forward pass
        if hasattr(self.model, "generator") and hasattr(self.model.generator, "decode"):
            return self.model.generator.decode(z)
        elif hasattr(self.model, "decode"):
            return self.model.decode(z)
        else:
            # Fallback to forward pass
            return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Returns the output shape for a given input shape.

        Args:
            input_shape: Input shape tuple (B, C, H, W)

        Returns:
            Output shape tuple (B, C_out, H, W)

        """
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "VQGANGenerator"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters.

        Returns:
            Total parameter count

        """
        return sum(p.numel() for p in self.parameters())

    def encode(self, x):
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Quantized latent tensor

        """
        if hasattr(self.model, "generator") and hasattr(self.model.generator, "encode"):
            return self.model.generator.encode(x)
        # Fallback if structure is different
        return self.model.encode(x) if hasattr(self.model, "encode") else x

    def decode(self, z):
        """Decode latent to image space.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed image tensor

        """
        if hasattr(self.model, "generator") and hasattr(self.model.generator, "decode"):
            return self.model.generator.decode(z)
        # Fallback if structure is different
        return self.model.decode(z) if hasattr(self.model, "decode") else z

    def get_discriminator(self):
        """Get discriminator module for adversarial training.

        Returns:
            Discriminator module

        """
        return self.model.discriminator if hasattr(self.model, "discriminator") else None

    @staticmethod
    def _init_weights(module):
        """Initialize weights using appropriate schemes.

        Args:
            module: Module to initialize

        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def create_vqgan_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    codebook_size: int = 1024,
    embedding_dim: int = 256,
    hiddein_channels: int = 128,
    num_embeddings: int = 1024,
    commitment_cost: float = 0.25,
    num_layers: int = 4,
    downsample_factor: int = 8,
    disc_channels: int = 64,
    disc_layers: int = 4,
    use_spectral_norm: bool = True,
    ema_decay: float = 0.999,
    reconstruction_weight: float = 1.0,
    perceptual_weight: float = 0.35,
    lpips_weight: float = 0.15,
    adversarial_weight: float = 0.05,
    **kwargs,
) -> VQGANGenerator:
    """Factory function to create VQGANGenerator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        codebook_size: Alias for num_embeddings (number of codebook entries)
        embedding_dim: Dimension of each codebook entry
        hiddein_channels: Number of hidden channels in encoder/decoder
        num_embeddings: Number of codebook entries (alternative parameter name)
        commitment_cost: Weight for commitment loss
        num_layers: Number of encoder/decoder layers
        downsample_factor: Total downsampling factor
        disc_channels: Number of discriminator channels
        disc_layers: Number of discriminator layers
        use_spectral_norm: Whether to use spectral normalization
        ema_decay: EMA decay rate for codebook updates
        reconstruction_weight: Weight for reconstruction loss
        perceptual_weight: Weight for perceptual loss
        lpips_weight: Weight for LPIPS loss
        adversarial_weight: Weight for adversarial loss
        **kwargs: Additional parameters passed to VQGANGenerator

    Returns:
        VQGANGenerator instance

    """
    return VQGANGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        hiddein_channels=hiddein_channels,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        num_layers=num_layers,
        downsample_factor=downsample_factor,
        disc_channels=disc_channels,
        disc_layers=disc_layers,
        use_spectral_norm=use_spectral_norm,
        ema_decay=ema_decay,
        reconstruction_weight=reconstruction_weight,
        perceptual_weight=perceptual_weight,
        lpips_weight=lpips_weight,
        adversarial_weight=adversarial_weight,
        **kwargs,
    )
