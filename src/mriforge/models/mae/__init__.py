"""MAE (Masked Autoencoder) Package
================================

This package contains the Masked Autoencoder implementation for self-supervised
pretraining on MRI volumes.
"""

from .mae_generator import MAEGenerator
from .simae_generator import SiMAEGenerator
from .simmlm_generator import SimMLMGenerator


def create_mae_generator(
    img_size: tuple = (64, 64, 64),
    patch_size: tuple = (16, 16, 16),
    in_channels: int = 1,
    embed_dim: int = 768,
    encoder_depth: int = 12,
    decoder_depth: int = 8,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    mask_ratio: float = 0.75,
) -> MAEGenerator:
    """Create a Masked Autoencoder generator for MRI volumes.

    Args:
        img_size: Input image size (H, W, D)
        patch_size: Patch size for tokenization (H, W, D)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        encoder_depth: Number of encoder transformer blocks
        decoder_depth: Number of decoder transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        mask_ratio: Ratio of patches to mask during pretraining

    Returns:
        MAEGenerator instance

    """
    return MAEGenerator(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        mask_ratio=mask_ratio,
    )


__all__ = ["MAEGenerator", "SiMAEGenerator", "SimMLMGenerator", "create_mae_generator"]
