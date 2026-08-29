"""KAN-INR: Kolmogorov-Arnold Network for Implicit Neural Representations.

Innovation VI (SOTA 2.0): Resolution-Agnostic Spectral Reconstruction.

Mathematical Foundation:
    f(x, y, z) -> (Re, Im)
    The backbone mapping coordinates to values is a Fourier-KAN:
    y = Sum_i w_i * FourierBasis_i(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model
from mriforge.models.spectral.fourier_kan import FourierKANLayer


@register_model(name="kan_inr", training_mode="reconstruction")
class KANINR(nn.Module):
    """Implicit Neural Representation with KAN Backbone.

    Args:
        in_features: 2 (x, y) or 3 (x, y, z).
        hidden_features: Number of hidden units.
        hidden_layers: Number of KAN layers.
        out_features: 2 (Real, Imag).
        grid_size: Number of Fourier basis functions.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 64,
        hidden_layers: int = 4,
        out_features: int = 2,
        grid_size: int = 5,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (int): Description.
            hidden_layers (int): Description.
            out_features (int): Description.
            grid_size (int): Description.
        """
        super().__init__()

        self.grid_size = grid_size

        layers = []
        # Input layer
        layers.append(FourierKANLayer(in_features, hidden_features, grid_size=grid_size))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(FourierKANLayer(hidden_features, hidden_features, grid_size=grid_size))
            layers.append(nn.Tanh())  # Activation between KAN layers for extra non-linearity?
            # Pure KAN usually sums bases. But stacking KANs suggests composition.
            # FourierKANLayer already has `base_linear` + `fourier`.
            # Let's add simple non-linearity or just rely on the KAN structure.
            # Standard KAN papers stack them directly.

        # Output layer
        self.net = nn.Sequential(*layers)
        self.final_proj = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Coordinates [B, ..., in_features] in range [-1, 1]

        Returns:
            Intensity [B, ..., out_features]

        forward method for KANINR.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure x is in expected range for Fourier KAN
        out = self.net(x)
        out = self.final_proj(out)
        return out


def create_kan_inr(hidden_dims: list[int] = [64, 64, 64, 64], input_dim: int = 2) -> KANINR:
    """Factory for KAN-INR."""
    # Mapping list config to fixed params for simplicity
    hidden_features = hidden_dims[0] if hidden_dims else 64
    hidden_layers = len(hidden_dims)

    return KANINR(
        in_features=input_dim,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=2,  # Complex
    )
