#!/usr/bin/env python3
"""Model Components Package
==================

This package contains shared model components and utilities
following SOLID principles.
"""

# Diffusion components
# Import from submodules
from ..blocks.embeddings import SinusoidalPositionEmbedding
from ..blocks.unet import DoubleConv, Down, Up
from ..diffusion.components import AttentionBlock
from ..transformer.components import FeedForward, MultiHeadSelfAttention, PatchEmbedding

__all__ = [
    # UNet components
    "DoubleConv",
    "Down",
    "Up",
    # Transformer components
    "PatchEmbedding",
    "MultiHeadSelfAttention",
    "FeedForward",
    # Diffusion components
    "SinusoidalPositionEmbedding",
    "AttentionBlock",
    "ConditioningEncoder",
]

from .conditioning_encoder import ConditioningEncoder
