"""Velocity Network for KIDOT
=============================

Neural network that predicts velocity fields for continuous normalizing
flow between VLF and HF distributions.

The velocity network v(x, t) defines the ODE:
    dx/dt = v(x, t)

Integration from t=0 to t=1 transports VLF → HF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model


@register_model(name="velocity_network", training_mode="reconstruction")
class VelocityNetwork(nn.Module):
    """Velocity network for continuous normalizing flow.

    Predicts velocity field v(x, t) that defines the transport ODE.

    Architecture: U-Net style encoder-decoder with time conditioning.

    Args:
        in_channels: Input image channels
        hidden_channels: Base hidden channels
        num_layers: Number of encoder/decoder layers
        time_embed_dim: Dimension of time embedding
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            hidden_channels (int): Description.
            num_layers (int): Description.
            time_embed_dim (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Encoder
        self.encoder = nn.ModuleList()
        ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_channels * (2 ** min(i, 2))
            self.encoder.append(ConvBlock(ch + (time_embed_dim if i == 0 else 0), out_ch))
            ch = out_ch

        # Bottleneck
        bottleneck_ch = ch
        self.bottleneck = ConvBlock(ch, ch)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = ch + hidden_channels * (2 ** min(i, 2))  # Skip connection
            out_ch = hidden_channels * (2 ** min(max(i - 1, 0), 2))
            if i == 0:
                out_ch = hidden_channels
            self.decoder.append(ConvBlock(in_ch, out_ch))
            ch = out_ch

        # Output: velocity field (same shape as input)
        self.output_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

        # Initialize output to near-zero
        self._init_output()

    def _init_output(self) -> None:
        """Initialize output layer for near-identity."""
        with torch.no_grad():
            self.output_conv.weight.fill_(0.0)
            self.output_conv.bias.fill_(0.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field.

        Args:
            x: Current state [B, C, H, W]
            t: Time [B] or [B, 1] in [0, 1]

        Returns:
            Velocity field [B, C, H, W]

        forward method for VelocityNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_emb = self.time_embed(t)  # [B, time_embed_dim]

        # Expand time embedding to spatial dimensions
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Concatenate time with input
        h = torch.cat([x, t_spatial], dim=1)

        # Encoder
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            if i == 0:
                h = layer(h)
            else:
                h = layer(h)
            skip_connections.append(h)
            if i < self.num_layers - 1:
                h = F.avg_pool2d(h, 2)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for i, layer in enumerate(self.decoder):
            skip_idx = self.num_layers - 1 - i
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            skip = skip_connections[skip_idx]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = layer(h)

        # Output velocity
        velocity = self.output_conv(h)

        return velocity


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h + self.residual(x))
        return h


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int):
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.dim = dim

        import math

        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim).float() / half_dim)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed time values.

        Args:
            t: Time [B, 1]

        Returns:
            Embedding [B, dim]

        forward method for SinusoidalEmbedding.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t = t * 1000.0  # Scale for frequency coverage
        args = t * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
