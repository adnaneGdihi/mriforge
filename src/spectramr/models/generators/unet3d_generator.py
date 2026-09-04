#!/usr/bin/env python3
"""3D UNet Generator Implementation
===============================

Concrete implementation of IGenerator for 3D UNet-based generators.
Supports volumetric medical imaging with full 3D convolutions.
Follows SOLID principles with single responsibility and dependency injection.
"""

from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.unet import Down3D, OutConv3D, Up3D
from spectramr.models.registry import register_model

from .base_3d_generator import Base3DGenerator, standardize_3d_params


@register_model(name="unet3d", training_mode="reconstruction")
class UNet3DGenerator(Base3DGenerator):
    """3D UNet-based generator implementation following SOLID principles.

    Single Responsibility: Only handles 3D UNet generation for volumetric data
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IGenerator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    __expected_kwargs__: set[str] = {
        "depth",
        "height",
        "width",
        "latent_shape",
        "output_dims",
        "device",
    }

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        trilinear: bool = True,
        output_activation: nn.Module | None = None,
        **kwargs,
    ):
        # Standardize parameters for backward compatibility while allowing
        # callers to override the default spatial dimensions.
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (tuple[int, ...]): Description.
            trilinear (bool): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        standardized_kwargs: dict[str, Any] = dict(kwargs)
        standardized_kwargs.setdefault("depth", 16)
        standardized_kwargs.setdefault("height", 16)
        standardized_kwargs.setdefault("width", 16)

        params: dict[str, Any] = standardize_3d_params(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            output_activation=output_activation,
            **standardized_kwargs,
        )

        super().__init__(**params)
        self.trilinear = trilinear

        # Encoder (Downsampling path) - 3 levels for 3D
        # to avoid excessive downsampling
        self.downs = nn.ModuleList()
        in_channels = self.in_channels
        for feature in self.features:
            self.downs.append(Down3D(in_channels, feature))
            in_channels = feature

        # Decoder (Upsampling path)
        self.ups = nn.ModuleList()
        # up blocks take concatenated channels as input
        self.ups.append(Up3D(384, 128, trilinear))  # 256+128=384
        self.ups.append(Up3D(192, 64, trilinear))  # 128+64=192
        self.ups.append(Up3D(96, 32, trilinear))  # 64+32=96
        self.ups.append(Up3D(64, 32, trilinear))  # 32+32=64

        # Output layer
        self.outc = OutConv3D(
            self.features[0],
            self.out_channels,
            activation=nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D UNet.

        Args:
            x: Input tensor [N, C, D, H, W] where N is batch size,
               C is number of channels, D, H, W are spatial dimensions

        Returns:
            Output tensor [N, C_out, D, H, W] preserving input
            spatial dimensions
        """
        # Encoder with skip connections
        x1 = self.downs[0](x)  # 32 channels
        x2 = self.downs[1](x1)  # 64 channels
        x3 = self.downs[2](x2)  # 128 channels
        x4 = self.downs[3](x3)  # 256 channels (bottleneck)

        # Decoder with skip connections
        x = self.ups[0](x4, x3)  # 384 -> 128
        x = self.ups[1](x, x2)  # 192 -> 64
        x = self.ups[2](x, x1)  # 96 -> 32
        x = self.ups[3](x, x1)  # 64 -> 32 (use x1 as skip for last layer)

        # Output layer
        logits = self.outc(x)
        return self.output_activation(logits)


def create_unet3d_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    features: tuple[int, ...] = (32, 64, 128, 256),
    trilinear: bool = True,
    output_activation: nn.Module | None = None,
    **kwargs,
) -> UNet3DGenerator:
    """Create a 3D UNet generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        features: Number of features at each level
        trilinear: Whether to use trilinear upsampling
        output_activation: Optional output activation function
        **kwargs: Additional arguments passed to UNet3DGenerator

    Returns:
        UNet3DGenerator instance

    """
    return UNet3DGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        trilinear=trilinear,
        output_activation=output_activation,
        **kwargs,
    )
