#!/usr/bin/env python3
"""DCSRN 3D Generator Implementation
=================================

Dense Channel Squeeze-and-Excitation Residual Network for 3D super-resolution.
Combines dense connections, channel attention, and residual learning for
volumetric medical imaging enhancement.

References:
- DenseNet: Densely Connected Convolutional Networks
- SENet: Squeeze-and-Excitation Networks
- EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution

"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.base.base_model import BaseGenerator
from spectramr.models.registry import register_model


class DenseBlock3D(nn.Module):
    """Dense block for 3D feature extraction."""

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int = 4):
        """__init__.

        Args:
            in_channels (int): Description.
            growth_rate (int): Description.
            num_layers (int): Description.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(DenseLayer3D(in_channels + i * growth_rate, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DenseBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseLayer3D(nn.Module):
    """Single dense layer with bottleneck and compression."""

    def __init__(self, in_channels: int, growth_rate: int):
        """__init__.

        Args:
            in_channels (int): Description.
            growth_rate (int): Description.
        """
        super().__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DenseLayer3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D."""

    def __init__(self, channels: int, reduction: int = 16):
        """__init__.

        Args:
            channels (int): Description.
            reduction (int): Description.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SEBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock3D(nn.Module):
    """Residual block with dense connections and SE attention."""

    def __init__(self, channels: int, growth_rate: int = 32, num_dense_layers: int = 4):
        """__init__.

        Args:
            channels (int): Description.
            growth_rate (int): Description.
            num_dense_layers (int): Description.
        """
        super().__init__()

        self.dense_block = DenseBlock3D(channels, growth_rate, num_dense_layers)
        self.se_block = SEBlock3D(channels + num_dense_layers * growth_rate)

        # 1x1x1 convolution to match channels
        self.conv1x1 = nn.Conv3d(
            channels + num_dense_layers * growth_rate,
            channels,
            1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResidualBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x
        out = self.dense_block(x)
        out = self.se_block(out)
        out = self.conv1x1(out)
        return out + residual


@register_model(name="dcsrn", training_mode="reconstruction")
class DCSRNGenerator(BaseGenerator):
    """Dense Channel Squeeze-and-Excitation Residual Network for 3D SR.

    Combines dense connections, channel attention, and residual learning
    for volumetric super-resolution in medical imaging.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_blocks: int = 16,
        growth_rate: int = 32,
        num_dense_layers: int = 4,
        scale_factor: int = 2,
        output_activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_blocks (int): Description.
            growth_rate (int): Description.
            num_dense_layers (int): Description.
            scale_factor (int): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        super().__init__(in_channels, out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.output_activation = output_activation or nn.Tanh()

        # Initial feature extraction
        self.head = nn.Conv3d(in_channels, 64, 3, padding=1)

        # Dense residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock3D(64, growth_rate, num_dense_layers) for _ in range(num_blocks)],
        )

        # Feature fusion
        self.fusion = nn.Conv3d(64, 64, 1)

        # Upsampling
        self.upsampler = self._make_upsampler(64, scale_factor)

        # Output
        self.tail = nn.Conv3d(64, out_channels, 3, padding=1)

    def _make_upsampler(self, channels: int, scale_factor: int) -> nn.Module:
        """Create upsampling module."""
        if scale_factor == 2:
            return nn.Sequential(
                nn.Conv3d(channels, channels * 8, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.PReLU(),
                nn.Conv3d(channels * 8, channels, 3, padding=1),
            )
        if scale_factor == 4:
            return nn.Sequential(
                nn.Conv3d(channels, channels * 8, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.PReLU(),
                nn.Conv3d(channels * 8, channels * 8, 3, padding=1),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.PReLU(),
                nn.Conv3d(channels * 8, channels, 3, padding=1),
            )
        raise ValueError(
            f"Unsupported upsample scale factor: {scale_factor}. Valid choices are: 2, 4"
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DCSRNGenerator"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DCSRN.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Super-resolved output

        """
        if x.dim() != 5:
            raise ValueError(
                f"DCSRNGenerator expects 5D input tensor [N, C, D, H, W], "
                f"got {x.dim()}D tensor with shape {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "DCSRNGenerator expects input with "
                + str(self.in_channels)
                + " channels, got "
                + str(x.shape[1])
                + " channels"
            )

        # Feature extraction
        head = self.head(x)

        # Dense residual processing
        body = self.body(head)
        body = self.fusion(body) + head  # Residual connection

        # Upsampling
        upsampled = self.upsampler(body)

        # Output
        out = self.tail(upsampled)

        return self.output_activation(out)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        n, _, d, h, w = input_shape
        scale = self.scale_factor
        return (n, self.out_channels, d * scale, h * scale, w * scale)


def create_dcsrn_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    num_blocks: int = 16,
    scale_factor: int = 2,
) -> DCSRNGenerator:
    """Create a DCSRN generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        num_blocks: Number of residual blocks
        scale_factor: Upsampling factor (2 or 4)

    Returns:
        DCSRNGenerator instance

    """
    return DCSRNGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        scale_factor=scale_factor,
    )
