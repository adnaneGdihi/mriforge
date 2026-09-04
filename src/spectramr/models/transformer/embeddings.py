"""Transformer Embeddings
======================

This module contains embedding layers for transformer models.
It re-exports canonical embeddings from spectramr.models.blocks.embeddings.
"""

from spectramr.models.blocks.embeddings import PatchEmbedding, SinusoidalPositionEmbedding

__all__ = [
    "PatchEmbedding",
    "SinusoidalPositionEmbedding",
]
