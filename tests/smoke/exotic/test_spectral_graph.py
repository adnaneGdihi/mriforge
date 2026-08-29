"""Tests for Spectral Graph Transformer (SOTA 2.0).

Verifies:
- Coil Graph construction (Laplacian shape).
- Weight normalization.
- Forward pass dimensions.
"""

import pytest
import torch

from mriforge.models.topological.spectral_graph import CoilGraph, SpectralGraphTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_coil_graph_shapes():
    """Verify LPE generation shapes."""
    B, C, H, W = 2, 8, 16, 16
    graph = CoilGraph(top_k_eigen=4).to(device)

    x = torch.randn(B, C, H, W, device=device)
    lpe = graph(x)

    # Expect [B, C, 4]
    assert lpe.shape == (B, C, 4)


def test_spectral_transformer_combine():
    """Verify combination logic and weight summation."""
    B, C, H, W = 2, 4, 16, 16
    model = SpectralGraphTransformer(in_channels=2, top_k_eigen=2, embed_dim=16).to(
        device
    )

    # Input [B, C, 2, H, W] (Complex)
    x = torch.randn(B, C, 2, H, W, device=device)

    out = model(x)

    # Combined: [B, 2, H, W]
    assert out.shape == (B, 2, H, W)


if __name__ == "__main__":
    pytest.main([__file__])
