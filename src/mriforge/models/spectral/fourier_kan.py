"""Fourier-KAN: Kolmogorov-Arnold Network with Fourier Basis.

Innovation IV (SOTA 2.0): Fourier-KAN for High-Frequency Recovery.

Mathematical Foundation:
    KAN Theorem: f(x) = sum_q Phi_q ( sum_p phi_{q,p} (x_p) )

    In Fourier-KAN, activation functions phi are learnable Fourier series:
    phi(x) = sum_{k=1}^K [ a_k sin(k pi x) + b_k cos(k pi x) ]

    Efficient Implementation:
    Instead of summing individual scalar functions, we use matrix operations:
    y = Linear(x) + FourierBasis(x) @ FourierCoeffs
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierKANLayer(nn.Module):
    """Dense layer with Fourier Basis activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        add_bias: bool = True,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            grid_size (int): Description.
            add_bias (bool): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size  # Number of frequencies

        # 1. Base Linear Layer (SiLU activation approximation / residual)
        self.base_linear = nn.Linear(in_features, out_features, bias=add_bias)

        # 2. Fourier Coefficients
        # Shape: [out, in, 2 * grid] -> but optimized
        # We want to compute: sum_{in} phi(x_in)
        # phi(x) has 2*grid parameters.
        # Total params: in * out * 2 * grid

        # We can implement as:
        # basis = [sin(1 pi x), cos(1 pi x), ..., sin(K pi x), cos(K pi x)] -> Shape [B, in, 2K]
        # output = basis * weights

        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, 2 * grid_size)
            / (math.sqrt(in_features) * grid_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., in_features]. Should be normalized to [-1, 1] essentially.

        forward method for FourierKANLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_shape = x.shape
        x_flat = x.view(-1, self.in_features)  # [N, In]

        # 1. Base Linear
        base_out = self.base_linear(x_flat)

        # 2. Fourier Basis
        # x is mapped to roughly [-1, 1].
        # k ranges from 1 to grid_size
        k = torch.arange(1, self.grid_size + 1, device=x.device).float()  # [K]
        k = k.view(1, 1, -1)  # [1, 1, K]

        x_expanded = x_flat.unsqueeze(-1)  # [N, In, 1]

        # Angles: k * pi * x
        angles = x_expanded * k * math.pi  # [N, In, K]

        # Basis components
        sin_basis = torch.sin(angles)  # [N, In, K]
        cos_basis = torch.cos(angles)  # [N, In, K]

        # Concatenate: [N, In, 2K]
        basis = torch.cat([sin_basis, cos_basis], dim=-1)

        # 3. Compute output
        # y[batch, out] = sum_{in} sum_{basis} weights[out, in, basis] * basis[batch, in, basis]
        # einsum: "n i b, o i b -> n o"
        kan_out = torch.einsum("nib,oib->no", basis, self.fourier_coeffs)

        # 4. Combine
        out = base_out + kan_out

        return out.view(*x_shape[:-1], self.out_features)


class KANBlock(nn.Module):
    """Feed-Forward Network using Fourier KAN Layers."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.0):
        """__init__.

        Args:
            in_dim (int): Description.
            hidden_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.layer1 = FourierKANLayer(in_dim, hidden_dim)
        self.layer2 = FourierKANLayer(hidden_dim, in_dim)
        self.act = nn.SiLU()  # KANs usually have activation on edges, but we can add mixing
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input to [-1, 1] range for Fourier stability?
        # KAN usually expects inputs in bounded range.
        # LayerNorm usually gives mean 0 std 1. Tanh map might be good.

        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_norm = torch.tanh(x)  # Soft clamp to region of interest

        x = self.layer1(x_norm)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(torch.tanh(x))
        x = self.dropout(x)
        return x
