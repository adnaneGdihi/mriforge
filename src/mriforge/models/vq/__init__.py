"""Vector Quantized VAE Package

This package contains implementations of Vector Quantized Variational
Autoencoders for learning discrete latent representations.
"""

from mriforge.models.interfaces.models import IGenerator

from .quantizer import VQQuantizer
from .vqvae import VQVAE


def create_vqvae_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    hiddein_channels: int = 128,
    num_embeddings: int = 512,
    embedding_dim: int = 64,
    commitment_cost: float = 0.25,
    num_layers: int = 4,
    downsample_factor: int = 4,
    **kwargs,
) -> IGenerator:
    """Factory function for creating VQ-VAE generators.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hiddein_channels: Number of hidden channels
        num_embeddings: Number of codebook entries
        embedding_dim: Dimension of embedding vectors
        commitment_cost: Commitment loss weight
        num_layers: Number of encoder/decoder layers
        downsample_factor: Spatial downsampling factor

    Returns:
        VQVAE instance implementing IGenerator interface

    """
    return VQVAE(
        in_channels=in_channels,
        out_channels=out_channels,
        hiddein_channels=hiddein_channels,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        num_layers=num_layers,
        downsample_factor=downsample_factor,
    )


__all__ = ["VQVAE", "VQQuantizer", "create_vqvae_generator"]
