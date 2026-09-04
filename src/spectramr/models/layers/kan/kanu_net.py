import torch
import torch.nn as nn
import torch.nn.functional as F

from .fastkanconv import FastKANConvLayer


class DoubleConv(nn.Module):
    """Standard CNN Block - High Efficiency"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kan_type=None):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            mid_channels (Any): Description.
            kan_type (Any): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Ignore kan_type, always use standard Conv2d for efficiency in Encoder/Decoder
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DoubleConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kan_type=None):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            kan_type (Any): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, kan_type=kan_type)
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Down.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bilinear=True):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            mid_channels (Any): Description.
            bilinear (Any): Description.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2):
        """forward.

        Args:
            x1 (Any): Description.
            x2 (Any): Description.
        Returns:
            Any: Description.

        forward method for Up.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """OutConv class."""

    def __init__(self, in_channels, out_channels):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Tanh())

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for OutConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.conv(x)


class KANBottleneck(nn.Module):
    """KAN Layer for Semantic Reasoning (High complexity, Low resolution)"""

    def __init__(self, in_channels, out_channels):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        # KAN is applied only here to learn non-linear semantic manifolds
        self.kan = FastKANConvLayer(
            in_channels, out_channels, kernel_size=3, padding=1, kan_type="RBF"
        )
        self.norm = nn.LayerNorm([out_channels, 16, 16])  # Assuming ~16x16 bottleneck

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for KANBottleneck.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.kan(x)


class KANU_Net(nn.Module):
    """KANU_Net class."""

    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            bilinear (Any): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.channels = [32, 64, 128]

        # 1. Spatial Encoder (Standard CNN)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        # 2. The Semantic Bridge (KAN)
        # We perform the KAN operation at the bottleneck only
        self.bottleneck_pool = nn.MaxPool2d(2)
        factor = 2 if bilinear else 1
        bottleneck_out_channels = 512 // factor

        # Explicitly use KANBottleneck here
        self.bottleneck_kan = nn.Sequential(KANBottleneck(256, bottleneck_out_channels))

        # 3. Spatial Decoder (Standard CNN with skip connections)
        self.up1 = Up(512, 256 // factor, bilinear=bilinear)
        self.up2 = Up(256, 128 // factor, bilinear=bilinear)
        self.up3 = Up(128, 64 // factor, bilinear=bilinear)
        self.up4 = Up(64, 32, bilinear=bilinear)

        # Final output convolution
        self.outc = OutConv(32, out_channels)

    def forward(self, x, encode_only=False, layers=None):
        # Encoder with skip connections
        """forward.

        Args:
            x (Any): Description.
            encode_only (Any): Description.
            layers (Any): Description.
        Returns:
            Any: Description.

        forward method for KANU_Net.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            encode_only (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            layers (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if encode_only:
            return [x1, x2, x3]

        # Bottleneck (KAN applied here only)
        x5 = self.bottleneck_pool(x4)
        x5 = self.bottleneck_kan(x5)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output
        return self.outc(x)
