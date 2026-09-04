"""Fourier Basis Functions for KAN Layers
=======================================

Fourier basis functions for Kolmogorov-Arnold Networks (KANs).
Provides sinusoidal basis functions for frequency domain feature learning.
"""

import math

import torch
from torch import nn


class FourierBasis(nn.Module):
    """Fourier basis functions for KAN activations."""

    def __init__(
        self,
        num_bases: int = 8,
        max_frequency: float = 1.0,
        trainable: bool = True,
        use_gabor: bool = False,
    ):
        """__init__.

        Args:
            num_bases (int): Description.
            max_frequency (float): Description.
            trainable (bool): Description.
            use_gabor (bool): Description.
        """
        super().__init__()
        self.num_bases = num_bases
        self.max_frequency = max_frequency
        self.use_gabor = use_gabor

        # Frequencies for Fourier basis
        if trainable:
            self.frequencies = nn.Parameter(
                torch.linspace(0.1, max_frequency, num_bases),
            )
        else:
            self.register_buffer(
                "frequencies",
                torch.linspace(0.1, max_frequency, num_bases),
            )

        # Phase shifts
        if trainable:
            self.phases = nn.Parameter(torch.zeros(num_bases))
            if self.use_gabor:
                self.means = nn.Parameter(torch.zeros(num_bases))
                self.variances = nn.Parameter(torch.ones(num_bases))
        else:
            self.register_buffer("phases", torch.zeros(num_bases))
            if self.use_gabor:
                self.register_buffer("means", torch.zeros(num_bases))
                self.register_buffer("variances", torch.ones(num_bases))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Fourier basis functions.

        forward method for FourierBasis.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # x shape: (batch, channels, height, width) or (batch, features)
        original_shape = x.shape

        # Flatten spatial dimensions if present
        if len(original_shape) > 2:
            x_flat = x.view(original_shape[0], -1)
        else:
            x_flat = x

        # Compute Fourier basis — broadcast over the basis dim instead of a
        # Python loop over ``num_bases`` (PERF 2026-07-01: the loop launched
        # 2*num_bases elementwise kernels + a stack per forward). The
        # multiplication grouping ``(2π·f)·x`` matches the old scalar path
        # exactly, so the result is bitwise identical per element.
        # arg: (batch, features, num_bases)
        arg = 2 * math.pi * self.frequencies * x_flat.unsqueeze(-1) + self.phases
        sin_component = torch.sin(arg)
        cos_component = torch.cos(arg)

        # Gabor envelope
        if self.use_gabor:
            envelope = torch.exp(-0.5 * ((x_flat.unsqueeze(-1) - self.means) / self.variances) ** 2)
            sin_component = sin_component * envelope
            cos_component = cos_component * envelope

        # Interleave to [sin_0, cos_0, sin_1, cos_1, ...] — the exact order
        # the old per-basis ``extend`` produced, so ``kan_weights`` layouts
        # and checkpoints stay compatible.
        # (batch, features, num_bases, 2) -> (batch, features, num_bases * 2)
        basis_tensor = torch.stack([sin_component, cos_component], dim=-1).flatten(-2)

        # Reshape to match input dimensions
        if len(original_shape) > 2:
            # Reshape back to spatial dimensions
            batch_size = original_shape[0]
            spatial_size = original_shape[1] * original_shape[2] * original_shape[3]
            basis_tensor = basis_tensor.view(batch_size, spatial_size, -1)
            basis_tensor = basis_tensor.view(
                batch_size,
                original_shape[1],
                original_shape[2],
                original_shape[3],
                -1,
            )

        return basis_tensor


from spectramr.models.layers.complex_conv import ComplexConv2d


class FourierKANConv2DLayer(nn.Module):
    """Convolutional layer with Fourier-based KAN activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_bases: int = 8,
        max_frequency: float = 1.0,
        basis_trainable: bool = True,
        use_complex_conv: bool = False,
        use_gabor: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            num_bases (int): Description.
            max_frequency (float): Description.
            basis_trainable (bool): Description.
            use_complex_conv (bool): Description.
            use_gabor (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bases = num_bases

        # Convolution
        if use_complex_conv:
            self.conv = ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        # Fourier basis for KAN activations
        self.fourier_basis = FourierBasis(
            num_bases=num_bases,
            max_frequency=max_frequency,
            trainable=basis_trainable,
            use_gabor=use_gabor,
        )

        # KAN weights for basis combination
        # Each output channel has weights for all basis functions
        total_basis = num_bases * 2  # sin and cos for each frequency
        self.kan_weights = nn.Parameter(torch.randn(out_channels, total_basis))

        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FourierKANConv2DLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        conv_out = self.conv(x)

        # Get Fourier basis functions
        # basis_out shape: (batch, out_channels, H, W, total_basis)
        basis_out = self.fourier_basis(conv_out)

        # KAN combination: weighted sum of basis functions
        # kan_weights shape: (out_channels, total_basis)
        # basis_out shape: (batch, out_channels, H, W, total_basis)
        # Output shape: (batch, out_channels, H, W)
        combined = torch.einsum("bohwt,ot->bohw", basis_out, self.kan_weights)

        # Add bias
        combined = combined + self.bias.view(1, -1, 1, 1)

        # Batch normalization
        output = self.bn(combined)

        return output
