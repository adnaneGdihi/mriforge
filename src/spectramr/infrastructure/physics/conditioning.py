"""Physics-Informed Conditioning Modules.
=======================================

This module implements conditioning mechanisms that incorporate MRI physics parameters
(such as acceleration factor) into the network architecture.
"""

import torch
import torch.nn as nn


class PhysicsInformedConditioning(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) conditioned on physics parameters.

    Modulates feature maps based on embeddings derived from physics parameters
    (e.g., time step, acceleration factor, noise level).

    y = x * (1 + scale) + shift
    """

    def __init__(self, embedding_dim: int, channel_dim: int):
        """
        Args:
            embedding_dim: Dimension of the input physics embedding.
            channel_dim: Number of channels in the feature map to modulate.
        """
        super().__init__()
        # Map embedding -> Scale & Shift parameters
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, channel_dim * 2),  # Output [scale, shift]
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, C, H, W]
            emb: Physics embedding [B, embedding_dim]

        Returns:
            Modulated feature map [B, C, H, W]

        forward method for PhysicsInformedConditioning.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Predict scale and shift
        params = self.fc(emb)
        scale, shift = params.chunk(2, dim=1)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1] (or [B, C, 1, 1, 1] for 5D)
        for _ in range(x.ndim - 2):
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)

        # Modulate
        # Note: We use (1 + scale) to allow zero-initialization of the linear layer
        # to result in identity modulation initially if weights were small,
        # though here we rely on standard init.
        return x * (1.0 + scale) + shift
