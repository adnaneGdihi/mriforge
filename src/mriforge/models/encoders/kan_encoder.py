"""KAN Encoder Implementation
==========================

This module provides a KAN (Kolmogorov-Arnold Network) based encoder
implementation using FastKANConvLayer.
"""

import torch
import torch.nn as nn

from mriforge.models.layers.kan.fastkanconv import FastKANConvLayer
from mriforge.models.registry import register_model


class KANBlock(nn.Module):
    """Basic KAN Block with convolution, normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        kan_type: str = "BSpline",
        norm_layer: nn.Module | None = nn.InstanceNorm2d,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            kan_type (str): Description.
            norm_layer (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.conv = FastKANConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            kan_type=kan_type,
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


@register_model(name="kan_encoder", training_mode="encoder")
class KANEncoder(nn.Module):
    """Encoder using KAN convolutions."""

    def __init__(
        self,
        in_channels: int = 1,
        features: list[int] = [32, 64, 128, 256],
        kan_type: str = "BSpline",
        norm_layer: nn.Module | None = nn.InstanceNorm2d,
        dropout: float = 0.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            features (list[int]): Description.
            kan_type (str): Description.
            norm_layer (Optional[nn.Module]): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.features = features
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        for feature in features:
            # Double KAN block
            block = nn.Sequential(
                KANBlock(
                    current_channels,
                    feature,
                    kan_type=kan_type,
                    norm_layer=norm_layer,
                ),
                KANBlock(
                    feature,
                    feature,
                    kan_type=kan_type,
                    norm_layer=norm_layer,
                ),
            )
            self.layers.append(block)

            # Downsample
            self.downsamples.append(nn.AvgPool2d(2))
            current_channels = feature

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            list[torch.Tensor]: Description.

        forward method for KANEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            list[torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)
            x = self.downsamples[i](x)
            x = self.dropout(x)

        return features
