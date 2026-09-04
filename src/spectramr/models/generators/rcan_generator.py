"""RCAN (Residual Channel Attention Network) Generator
==================================================

RCAN generator implementation for super-resolution tasks.
Based on the paper "Image Super-Resolution Using Very Deep
Residual Channel Attention Networks" by Zhang et al. (2018).

Features:
- Residual channel attention blocks (RCAB)
- Channel attention mechanism
- Deep residual architecture
- Excellent performance for SR tasks
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class ChannelAttention(nn.Module):
    """Channel Attention module for RCAB."""

    def __init__(self, num_features: int, reduction: int = 16):
        """__init__.

        Args:
            num_features (int): Description.
            reduction (int): Description.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ChannelAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, num_features: int, reduction: int = 16, skip_scale: float = 1.0):
        """__init__.

        Args:
            num_features (int): Description.
            reduction (int): Description.
            skip_scale (float): Description.
        """
        super().__init__()
        self.skip_scale = skip_scale
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(num_features, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RCAB.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.ca(out) * out
        out.mul_(self.skip_scale)
        out += residual
        return out


class RG(nn.Module):
    """Residual Group containing multiple RCABs."""

    def __init__(
        self,
        num_features: int,
        num_blocks: int,
        reduction: int = 16,
        skip_scale: float = 1.0,
    ):
        """__init__.

        Args:
            num_features (int): Description.
            num_blocks (int): Description.
            reduction (int): Description.
            skip_scale (float): Description.
        """
        super().__init__()
        self.rcabs = nn.ModuleList(
            [RCAB(num_features, reduction, skip_scale=skip_scale) for _ in range(num_blocks)],
        )
        self.conv = nn.Conv2d(num_features, num_features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RG.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x
        for rcab in self.rcabs:
            residual = rcab(residual)
        residual = self.conv(residual)
        residual += x
        return residual


@register_model(name="rcan", training_mode="reconstruction")
class RCANGenerator(IGenerator, nn.Module):
    """RCAN (Residual Channel Attention Network) generator for super-resolution.

    Features:
    - Multiple residual groups with channel attention
    - Long skip connections
    - Efficient architecture for SR tasks
    - State-of-the-art performance

    Args:
        in_channels: Number of input channels (default: 1 for grayscale MRI)
        out_channels: Number of output channels (default: 1)
        num_features: Number of feature maps (default: 64)
        num_groups: Number of residual groups (default: 10)
        num_blocks: Number of RCABs per group (default: 20)
        reduction: Channel attention reduction factor (default: 16)
        scale_factor: Upsampling scale factor (default: 2)
        output_activation: Output activation function

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_groups: int = 10,
        num_blocks: int = 20,
        reduction: int = 16,
        scale_factor: int = 2,
        skip_scale: float = 1.0,
        output_activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_features (int): Description.
            num_groups (int): Description.
            num_blocks (int): Description.
            reduction (int): Description.
            scale_factor (int): Description.
            skip_scale (float): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.reduction = reduction
        self.scale_factor = scale_factor
        self.skip_scale = torch.clamp(torch.tensor(skip_scale), 0.1, 1.0).item()
        self.output_activation = output_activation or nn.Identity()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels, num_features, 3, padding=1)

        # Residual groups
        self.rgs = nn.ModuleList(
            [
                RG(
                    num_features,
                    num_blocks,
                    reduction,
                    skip_scale=self.skip_scale,
                )
                for _ in range(num_groups)
            ],
        )

        # Global feature fusion
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        if scale_factor == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )
        elif scale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )
        else:
            # For other scale factors, use bilinear upsampling
            self.upsampler = nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                ),
                nn.Conv2d(num_features, out_channels, 3, padding=1),
            )

    @property
    def name(self) -> str:
        """Return model name."""
        return f"RCANGenerator_{self.num_groups}groups_{self.num_blocks}blocks_{self.scale_factor}x"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RCAN generator.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, out_channels, H*scale_factor, W*scale_factor]

        forward method for RCANGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initial feature extraction
        feat = self.conv1(x)

        # Residual groups with long skip connection
        residual = feat
        for rg in self.rgs:
            residual = rg(residual)
        residual = self.conv2(residual)
        residual += feat  # Long skip connection

        # Upsampling
        out = self.upsampler(residual)

        # Apply output activation
        return self.output_activation(out)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved image from low-resolution input.

        Args:
            z: Low-resolution input image
            **kwargs: Additional generation parameters

        Returns:
            Super-resolved output image

        """
        return self.forward(z)

    def get_output_shape(self, input_shape):
        """Calculate output shape for given input shape.

        Args:
            input_shape: Input tensor shape (N, C, H, W)

        Returns:
            Output tensor shape (N, out_channels, H*scale_factor, W*scale_factor)

        """
        n, _, h, w = input_shape
        return (n, self.out_channels, h * self.scale_factor, w * self.scale_factor)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["RCAB", "RG", "ChannelAttention", "RCANGenerator"]
