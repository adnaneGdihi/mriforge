"""Latent Architecture Module
==========================

Flexible latent space architectures for VAE, VQ-VAE, and diffusion models.
"""

from .configurable_architecture import (
    DecoderConfig,
    EncoderConfig,
    FlexibleDecoder,
    FlexibleEncoder,
    LatentConfig,
    ResBlock,
    SelfAttention2D,
    VectorQuantizer,
)

__all__ = [
    "DecoderConfig",
    "EncoderConfig",
    "FlexibleDecoder",
    "FlexibleEncoder",
    "LatentConfig",
    "ResBlock",
    "SelfAttention2D",
    "VectorQuantizer",
]
