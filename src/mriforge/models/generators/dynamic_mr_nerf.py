"""Dynamic MR-NeRF — Marker-Conditioned 4D Implicit Neural Representation.

Represents the entire moving MRI sequence as a single continuous function
``I = MLP(x, y, m(t))``, eliminating discrete frame binning and temporal
blurring entirely.

Architecture:
    .. code-block:: text

        (x, y) → Positional Encoding → spatial features
        m(t)   → Temporal Encoder     → temporal features
                     ↓
             Concatenate → MLP → Complex intensity (Re, Im)

Physics:
    Optimised directly against raw k-space via differentiable FFT:

    .. math::

        \\mathcal{L}_{DC} = \\|\\mathcal{F}[\\text{MLP}(\\mathbf{r}, m(t))]
            \\cdot M - y_{meas}\\|_2^2

Reference:
    Zhuang et al., "Cardiac NeRF: MRI-based 3D Dynamic Heart Modeling,"
    *Medical Image Analysis*, 2023.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for implicit representations.

    Maps low-dimensional coordinates to high-dimensional features via:
    ``[sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x)]``

    Args:
        input_dim: Coordinate dimension (2 for 2-D, 3 for 3-D).
        num_frequencies: Number of frequency octaves L.
    """

    def __init__(self, input_dim: int = 2, num_frequencies: int = 10) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * (2 * num_frequencies + 1)

        freqs = 2.0 ** torch.arange(num_frequencies) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode coordinates.

        Args:
            coords: ``[..., input_dim]``.

        Returns:
            Encoded features ``[..., output_dim]``.

        forward method for _PositionalEncoding.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # coords: [..., D], freqs: [L]
        # Outer product: [..., D, L]
        scaled = coords.unsqueeze(-1) * self.freqs  # [..., D, L]
        # Flatten and concat: [..., D*(2L+1)]
        encoded = torch.cat([coords, scaled.sin().flatten(-2), scaled.cos().flatten(-2)], dim=-1)
        return encoded


class _TemporalEncoder(nn.Module):
    """Encode marker signal m(t) into temporal conditioning features."""

    def __init__(self, marker_dim: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(marker_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, marker: Tensor) -> Tensor:
        """forward method for _TemporalEncoder.

        Executes PyTorch tensor operations.

        Args:
            marker (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker.dim() == 1:
            marker = marker.unsqueeze(-1)
        return self.net(marker)


@register_model(name="dynamic_mr_nerf", training_mode="virtual_fiducial")
class DynamicMRNeRF(nn.Module, IGenerator):
    """4-D implicit representation conditioned on marker signal.

    Represents the entire MRI sequence as a continuous function, queried
    at arbitrary spatial coordinates and time points.

    Args:
        in_channels: Input spatial dimension (2 for 2-D).
        out_channels: Output channels (2 for complex: Re, Im).
        marker_dim: Marker signal dimension.
        num_frequencies: Positional encoding octaves.
        hidden_dim: MLP hidden dimension.
        num_layers: Number of MLP hidden layers.
        im_size: Image size for grid-based rendering.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        marker_dim: int = 1,
        num_frequencies: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 6,
        im_size: tuple[int, int] = (256, 256),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.im_size = im_size

        # Positional encoding for spatial coords (always 2 for 2D images)
        self.pos_enc = _PositionalEncoding(input_dim=2, num_frequencies=num_frequencies)

        # Temporal encoder for marker signal
        temporal_dim = 64
        self.temp_enc = _TemporalEncoder(marker_dim, temporal_dim)

        # MLP: spatial_features + temporal_features → intensity
        input_dim = self.pos_enc.output_dim + temporal_dim
        layers: list[nn.Module] = []
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])

        for i in range(num_layers - 1):
            # Skip connection at midpoint
            if i == num_layers // 2:
                layers.extend([nn.Linear(hidden_dim + input_dim, hidden_dim), nn.GELU()])
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])

        layers.append(nn.Linear(hidden_dim, out_channels))
        self.mlp = nn.ModuleList(layers)

        # Store skip connection index
        self._skip_idx = 2 + (num_layers // 2) * 2  # after input + skip_layer pairs

        # Build normalised coordinate grid
        H, W = im_size
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([X, Y], dim=-1)  # [H, W, 2]
        self.register_buffer("_grid", coords)

        logger.info(
            "[DynamicMRNeRF] im=%s, freq=%d, hidden=%d, layers=%d",
            im_size,
            num_frequencies,
            hidden_dim,
            num_layers,
        )

    @property
    def name(self) -> str:
        return "dynamic_mr_nerf"

    def query(
        self,
        coords: Tensor,
        marker_signal: Tensor,
    ) -> Tensor:
        """Query the implicit representation at given coordinates.

        Args:
            coords: Spatial coordinates ``[B, N, 2]``.
            marker_signal: Marker signal ``[B]`` or ``[B, D]``.

        Returns:
            Predicted intensity ``[B, N, out_channels]``.
        """
        B, N, _ = coords.shape

        # Encode spatial coordinates
        spatial = self.pos_enc(coords)  # [B, N, pos_dim]

        # Encode temporal signal and expand to match spatial points
        temporal = self.temp_enc(marker_signal)  # [B, temporal_dim]
        temporal = temporal.unsqueeze(1).expand(-1, N, -1)  # [B, N, temporal_dim]

        # Concatenate
        features = torch.cat([spatial, temporal], dim=-1)  # [B, N, input_dim]
        skip_input = features

        # Forward through MLP with skip connection
        h = features
        layer_idx = 0
        for module in self.mlp:
            if isinstance(module, nn.Linear) and module.in_features > h.shape[-1]:
                # Skip connection layer
                h = torch.cat([h, skip_input], dim=-1)
            h = module(h)
            layer_idx += 1

        return h  # [B, N, out_channels]

    def render_image(
        self,
        marker_signal: Tensor,
    ) -> Tensor:
        """Render a full image at the stored grid coordinates.

        Args:
            marker_signal: ``[B]`` or ``[B, D]``.

        Returns:
            Image ``[B, out_channels, H, W]``.
        """
        B = marker_signal.shape[0] if marker_signal.dim() > 0 else 1
        H, W = self.im_size

        coords = self._grid.reshape(1, H * W, 2).expand(B, -1, -1)
        output = self.query(coords, marker_signal)  # [B, H*W, C]
        return output.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

    def forward(
        self,
        x: Tensor,
        marker_signal: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Forward pass: render image conditioned on marker.

        Args:
            x: Input tensor ``[B, C, H, W]`` (used for shape/device only).
            marker_signal: ``[B]`` marker signal.

        Returns:
            Reconstructed image ``[B, out_channels, H, W]``.

        forward method for DynamicMRNeRF.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_signal (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]
        if marker_signal is None:
            marker_dim = self.temp_enc.net[0].in_features
            marker_signal = torch.zeros(B, marker_dim, device=x.device)

        return self.render_image(marker_signal)

    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *self.im_size)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["DynamicMRNeRF"]
