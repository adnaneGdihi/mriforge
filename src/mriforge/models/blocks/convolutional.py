"""Convolutional Building Blocks
============================

Common convolutional blocks used across different model architectures.
"""

import math

import torch
from torch import nn

from mriforge.models.layers.complex_conv import ComplexConv2d


class ConvBlock2D(nn.Module):
    """Basic 2D convolutional block with optional normalization
    and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        norm_layer: nn.Module | None = None,
        activation: nn.Module | None = nn.ReLU(inplace=True),
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            bias (bool): Description.
            norm_layer (Optional[nn.Module]): Description.
            activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ConvBlock2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DoubleConv(nn.Module):
    """Double convolution block for U-Net with GroupNorm and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (Optional[int]): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(math.gcd(32, mid_channels), mid_channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(math.gcd(32, out_channels), out_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


from mriforge.models.blocks.normalization import AdaptiveGroupNorm


class ComplexDoubleConv(nn.Module):
    """Double convolution block for U-Net dealing with complex-valued data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (Optional[int]): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # We process Real/Imag channels as pairs.
        # If in_channels=2 (1 complex), we split.

        self.double_conv = nn.Sequential(
            ComplexConv2d(
                in_channels // 2,
                mid_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            ComplexConv2d(
                mid_channels // 2,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ComplexDoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class AdaGNDoubleConv(nn.Module):
    """Double convolution block with Adaptive Group Normalization (AdaGN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            emb_channels (int): Description.
            mid_channels (Optional[int]): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = AdaptiveGroupNorm(mid_channels, math.gcd(32, mid_channels), emb_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = AdaptiveGroupNorm(out_channels, math.gcd(32, out_channels), emb_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # AdaGN placement: norm -> cond -> act -> conv
        # But here we have a double conv block.
        # The standard pattern for AdaGN in a ResNet block is:
        # h = norm(x)
        # h = cond(h, emb)
        # h = act(h)
        # h = conv(h)
        # However, this is a DoubleConv block, not a PreAct ResBlock.
        # If we follow the standard DoubleConv pattern (conv -> norm -> act),
        # we apply AdaGN as the norm layer.
        # Let's keep it as conv -> norm(with cond) -> act to match the original intent of this block,
        # but note that for strict invertibility in diffusion, pre-act (norm->cond->act->conv) is preferred.
        # Since this is just a DoubleConv, we'll leave it as is, but ensure the norm itself applies the conditioning correctly.
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for AdaGNDoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv1(x)
        x = self.norm1(x, emb)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, emb)
        x = self.act2(x)
        return x


class ComplexAdaGNDoubleConv(nn.Module):
    """Complex Double convolution block with Adaptive Group Normalization (AdaGN)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            emb_channels (int): Description.
            mid_channels (Optional[int]): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = ComplexConv2d(
            in_channels // 2,
            mid_channels // 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = AdaptiveGroupNorm(mid_channels, math.gcd(32, mid_channels), emb_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = ComplexConv2d(
            mid_channels // 2,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = AdaptiveGroupNorm(out_channels, math.gcd(32, out_channels), emb_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x is (B, 2*Cin, H, W)
        # Similar to AdaGNDoubleConv, we apply conv -> norm(with cond) -> act
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ComplexAdaGNDoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv1(x)
        # Output is (B, 2*Cmid, H, W). Norm treats it as 2*Cmid channels.
        x = self.norm1(x, emb)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, emb)
        x = self.act2(x)
        return x


class ResConvBlock(nn.Module):
    """Residual convolutional block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            1,  # stride=1 for second conv
            padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResConvBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ConvBlock3D(nn.Module):
    """Basic 3D convolutional block with optional normalization
    and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        norm_layer: nn.Module | None = None,
        activation: nn.Module | None = nn.ReLU(inplace=True),
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            bias (bool): Description.
            norm_layer (Optional[nn.Module]): Description.
            activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ConvBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DoubleConv3D(nn.Module):
    """Double 3D convolution block commonly used in 3D U-Net architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (Optional[int]): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DoubleConv3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class ResConvBlock3D(nn.Module):
    """Residual 3D convolutional block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size,
            1,  # stride=1 for second conv
            padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm3d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ResConvBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


__all__ = [
    "ConvBlock2D",
    "ConvBlock3D",
    "DoubleConv",
    "DoubleConv3D",
    "ResConvBlock",
    "ResConvBlock3D",
]
