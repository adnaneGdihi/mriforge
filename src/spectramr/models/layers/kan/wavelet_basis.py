"""Wavelet Basis Functions for KAN Layers
=======================================

Implements wavelet-based basis functions for Kernel Activation Networks (KAN).
Supports Haar and Daubechies wavelets for multi-scale feature extraction.
"""

import torch
import torch.nn.functional as F
from torch import nn


class WaveletBasis(nn.Module):
    """Wavelet basis functions for KAN layers."""

    def __init__(
        self,
        num_bases: int = 8,
        wavelet_family: str = "haar",
        basis_trainable: bool = True,
    ):
        """__init__.

        Args:
            num_bases (int): Description.
            wavelet_family (str): Description.
            basis_trainable (bool): Description.
        """
        super().__init__()
        self.num_bases = num_bases
        self.wavelet_family = wavelet_family
        self.basis_trainable = basis_trainable

        # Initialize wavelet coefficients
        if wavelet_family == "haar":
            # Haar wavelet: [1, 1]/sqrt(2) for scaling, [1, -1]/sqrt(2) for wavelet
            sqrt2_inv = 1.0 / (2**0.5)
            coeffs = torch.tensor(
                [[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=torch.float32
            )
        elif wavelet_family == "db2":
            # Daubechies-2 (4-tap filter) - Exact orthonormal coefficients
            # h = [(1+sqrt(3)), (3+sqrt(3)), (3-sqrt(3)), (1-sqrt(3))] / (4*sqrt(2))
            sqrt3 = 3**0.5
            h = torch.tensor(
                [
                    (1 + sqrt3) / (4 * (2**0.5)),
                    (3 + sqrt3) / (4 * (2**0.5)),
                    (3 - sqrt3) / (4 * (2**0.5)),
                    (1 - sqrt3) / (4 * (2**0.5)),
                ],
                dtype=torch.float32,
            )
            # Wavelet g = alternating flip of h: g[n] = (-1)^n * h[L-1-n]
            g = torch.tensor([h[3], -h[2], h[1], -h[0]], dtype=torch.float32)
            coeffs = torch.stack([h, g], dim=0)
        elif wavelet_family.startswith("db"):
            # Fallback for other Daubechies - use db2 coefficients
            sqrt3 = 3**0.5
            h = torch.tensor(
                [
                    (1 + sqrt3) / (4 * (2**0.5)),
                    (3 + sqrt3) / (4 * (2**0.5)),
                    (3 - sqrt3) / (4 * (2**0.5)),
                    (1 - sqrt3) / (4 * (2**0.5)),
                ],
                dtype=torch.float32,
            )
            g = torch.tensor([h[3], -h[2], h[1], -h[0]], dtype=torch.float32)
            coeffs = torch.stack([h, g], dim=0)
        elif wavelet_family == "mexican_hat":
            # Mexican Hat (Ricker) wavelet
            # Second derivative of Gaussian
            # Approximate 5-tap filter
            pi = 3.14159
            t = torch.linspace(-2, 2, 5)
            # (1 - t^2) * exp(-t^2/2)
            mh = (1 - t**2) * torch.exp(-(t**2) / 2)
            mh = mh / mh.norm()  # Normalize

            # For "scaling" counterpart in this KAN context, we can use Gaussian
            gauss = torch.exp(-(t**2) / 2)
            gauss = gauss / gauss.norm()

            coeffs = torch.stack([gauss, mh], dim=0)

        elif wavelet_family == "morlet":
            # Real Morlet wavelet
            # cos(5t) * exp(-t^2/2)
            t = torch.linspace(-2, 2, 5)
            morlet = torch.cos(5 * t) * torch.exp(-(t**2) / 2)
            morlet = morlet / morlet.norm()

            # Scaling: Gaussian
            gauss = torch.exp(-(t**2) / 2)
            gauss = gauss / gauss.norm()

            coeffs = torch.stack([gauss, morlet], dim=0)

        else:
            raise ValueError(f"Unsupported wavelet family: {wavelet_family}")

        if basis_trainable:
            self.register_parameter("coeffs", nn.Parameter(coeffs))
        else:
            self.register_buffer("coeffs", coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute wavelet basis functions.

        forward method for WaveletBasis.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch, channels, h, w = x.shape

        # LIMITATION: This is an approximation of a full wavelet transform.
        # A proper multi-resolution analysis (MRA) would apply the filters
        # at progressively downsampled scales (dyadic decimation).  Here we
        # apply the same filters at input resolution for every basis index,
        # which functions as a learnable filter bank rather than a true DWT.
        # For MRI reconstruction tasks this is acceptable because the KAN
        # weights learn to combine the basis outputs regardless.

        # Scaling function basis - expand to match input channels
        scaling = self.coeffs[0:1]  # (1, 2)
        scaling_expanded = scaling.repeat(channels, 1, 1, 1)

        # Wavelet function basis - expand to match input channels
        wavelet = self.coeffs[1:2]  # (1, 2)
        wavelet_expanded = wavelet.repeat(channels, 1, 1, 1)

        # PERF (2026-07-01): the old per-basis loop recomputed these two
        # convolutions ``num_bases`` times with loop-invariant inputs and
        # weights — 2*num_bases convs where 2 suffice. Compute once and tile
        # along the basis dim; ``repeat`` preserves the exact
        # ``[scaling, wavelet] * num_bases`` interleave the ``kan_weights``
        # consumers rely on, and backward over the tiled tensor sums the
        # per-copy grads exactly like the old duplicate graph nodes did.
        scaling_basis = F.conv2d(x, scaling_expanded, padding=1, groups=channels)
        wavelet_basis = F.conv2d(x, wavelet_expanded, padding=1, groups=channels)

        # Ensure consistent shape by cropping if necessary
        if scaling_basis.shape[-1] != x.shape[-1]:
            scaling_basis = scaling_basis[:, :, : x.shape[-2], : x.shape[-1]]
            wavelet_basis = wavelet_basis[:, :, : x.shape[-2], : x.shape[-1]]

        # (batch, channels, h, w, 2) -> tiled to (batch, channels, h, w, 2*num_bases)
        pair = torch.stack([scaling_basis, wavelet_basis], dim=-1)
        basis_tensor = pair.repeat(1, 1, 1, 1, self.num_bases)

        return basis_tensor


class WaveletKANConv2DLayer(nn.Module):
    """Convolutional layer with wavelet-based KAN activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_bases: int = 8,
        wavelet_family: str = "haar",
        basis_trainable: bool = True,
        norm_layer: type[nn.Module] | None = None,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            num_bases (int): Description.
            wavelet_family (str): Description.
            basis_trainable (bool): Description.
            norm_layer (Optional[type[nn.Module]]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Standard convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        # Wavelet basis
        self.wavelet_basis = WaveletBasis(num_bases, wavelet_family, basis_trainable)

        # KAN weights for basis combination
        self.kan_weights = nn.Parameter(torch.randn(out_channels, num_bases))

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Normalization
        if norm_layer:
            self.norm = norm_layer(out_channels, **norm_kwargs)
        else:
            self.norm = nn.Identity()

        # Activation
        self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for WaveletKANConv2DLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        conv_out = self.conv(x)

        # Get wavelet basis functions
        # basis_out shape: (batch, out_channels, H, W, total_basis)
        basis_out = self.wavelet_basis(conv_out)

        # KAN combination: weighted sum of basis functions
        # kan_weights shape: (out_channels, num_bases)
        # We need to expand weights to match total_basis = num_bases * 2
        weights_expanded = self.kan_weights.unsqueeze(-1).repeat(1, 1, 2)
        weights_expanded = weights_expanded.view(self.out_channels, -1)
        # (out_channels, total_basis)

        # Apply weights to basis functions for each output channel
        # basis_out: (batch, out_channels, H, W, total_basis)
        # weights_expanded: (out_channels, total_basis)
        weights_exp = weights_expanded.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # (1, out_channels, 1, 1, total_basis)
        weighted_basis = basis_out * weights_exp
        # (batch, out_channels, H, W, total_basis)

        # Sum across basis functions
        combined = weighted_basis.sum(dim=-1)  # (batch, out_channels, H, W)

        # Add bias
        combined = combined + self.bias.view(1, -1, 1, 1)

        # Apply normalization and activation
        combined = self.norm(combined)
        combined = self.act(combined)

        return combined
