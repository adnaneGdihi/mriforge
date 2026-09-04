"""UNet-specific building blocks
=============================

Reusable UNet blocks extracted from generator implementations to
avoid duplication and keep files small and focused.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .convolutional import DoubleConv, DoubleConv3D


class Down(nn.Module):
    """Downsampling block - maxpool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Down.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block - upsample then DoubleConv with skip concat."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Up.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)

        # Optimization: Center Crop x2 to match x1 (Zero-Copy View)
        # Assumption: x2 is larger than x1 (standard in UNets without padding)
        if x1.shape != x2.shape:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output layer with 1x1 conv and optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for OutConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.activation(self.conv(x))


class Down3D(nn.Module):
    """3D Downsampling block - maxpool followed by DoubleConv3D."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Down3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """3D Upsampling block - upsample then DoubleConv3D with skip concat."""

    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            trilinear (bool): Description.
        """
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Up3D.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """3D Output layer with 1x1 conv and optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for OutConv3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.activation(self.conv(x))


__all__ = ["Down", "Down3D", "OutConv", "OutConv3D", "Up", "Up3D"]
