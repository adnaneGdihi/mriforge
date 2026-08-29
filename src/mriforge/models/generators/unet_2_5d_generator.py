"""2.5D UNet Generator
==================

Implementation of a 2.5D UNet generator for volumetric MRI super-resolution.
This model processes stacked 2D slices to capture 3D spatial context while
maintaining computational efficiency.

The architecture uses:
- 2D convolutions on stacked slices
- 3D context through slice stacking
- Efficient memory usage compared to full 3D models
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.registry import register_model


class DoubleConv2d(nn.Module):
    """Double 2D convolution block."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DoubleConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.double_conv(x)


class Down2d(nn.Module):
    """Downsampling block for 2D UNet."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Down2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.maxpool_conv(x)


class Up2d(nn.Module):
    """Upsampling block for 2D UNet."""

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
            self.conv = DoubleConv2d(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Up2d.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2d(nn.Module):
    """Output convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for OutConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.conv(x)


@register_model(name="unet_2_5d", training_mode="reconstruction")
class UNet2_5DGenerator(nn.Module):
    """2.5D UNet Generator for volumetric super-resolution.

    This generator processes stacked 2D slices to capture 3D context
    while using 2D convolutions for computational efficiency.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_slices: int = 5,
        features: tuple[int, ...] = (64, 128, 256, 512, 1024),
        bilinear: bool = True,
        scale_factor: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_slices (int): Description.
            features (tuple[int, ...]): Description.
            bilinear (bool): Description.
            scale_factor (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slices = num_slices
        self.scale_factor = scale_factor
        self.bilinear = bilinear

        # Adjust input channels for stacked slices
        effective_iin_channels = in_channels * num_slices

        # Encoder
        self.inc = DoubleConv2d(effective_iin_channels, features[0])
        self.down1 = Down2d(features[0], features[1])
        self.down2 = Down2d(features[1], features[2])
        self.down3 = Down2d(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down2d(features[3], features[4] // factor)

        # Decoder
        self.up1 = Up2d(features[4], features[3] // factor, bilinear)
        self.up2 = Up2d(features[3], features[2] // factor, bilinear)
        self.up3 = Up2d(features[2], features[1] // factor, bilinear)
        self.up4 = Up2d(features[1], features[0], bilinear)
        self.outc = OutConv2d(features[0], out_channels)

        # Additional upsampling for super-resolution
        if scale_factor > 1:
            self.final_upsample = self._make_final_upsampler(out_channels, scale_factor)

    def _make_final_upsampler(self, channels: int, scale_factor: int) -> nn.Module:
        """Create final upsampling module for super-resolution."""
        if scale_factor == 4:
            return nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        if scale_factor == 2:
            return nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        raise ValueError(f"Unsupported scale factor: {scale_factor}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 2.5D UNet.

        Args:
            x: Input tensor of shape (B, C, D, H, W) where D is depth/slices

        Returns:
            Super-resolved output

        """
        batch_size, channels, depth, height, width = x.shape

        # Stack slices for 2.5D processing
        # For each spatial position, stack neighboring slices
        if depth >= self.num_slices:
            # Use sliding window of slices
            stacked_features = []
            for i in range(self.num_slices):
                slice_idx = torch.clamp(
                    torch.arange(depth) + i - self.num_slices // 2,
                    0,
                    depth - 1,
                )
                stacked_slice = x[:, :, slice_idx, :, :]  # (B, C, D, H, W)
                # Average across the slice dimension to get (B, C, H, W)
                stacked_slice = stacked_slice.mean(dim=2)
                stacked_features.append(stacked_slice)

            # Concatenate along channel dimension: (B, C*num_slices, H, W)
            x_2d = torch.cat(stacked_features, dim=1)
        else:
            # If fewer slices than needed, repeat the available slices
            x_2d = x.squeeze(2)  # Remove depth dim if single slice
            if x_2d.dim() == 4:  # (B, C, H, W)
                x_2d = x_2d.repeat(1, self.num_slices, 1, 1)

        # UNet forward pass
        x1 = self.inc(x_2d)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # Apply final upsampling for super-resolution
        if self.scale_factor > 1:
            x = self.final_upsample(x)

        return x

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved output from 2.5D input."""
        return self.forward(x)

    @property
    def name(self) -> str:
        """Return model name."""
        return f"UNet2_5DGenerator_{self.num_slices}slices"

    def get_parameter_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape given input shape."""
        if len(input_shape) == 5:
            # 5D input: (batch_size, channels, depth, height, width)
            batch_size, channels, depth, height, width = input_shape
        elif len(input_shape) == 4:
            # 4D input: (batch_size, channels, height, width) - assume depth=1
            batch_size, channels, height, width = input_shape
        else:
            raise ValueError(f"Expected 4D or 5D input shape, got {len(input_shape)}D")
        return (
            batch_size,
            self.out_channels,
            height * self.scale_factor,
            width * self.scale_factor,
        )


# Create function for factory pattern
def create_unet_2_5d_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    num_slices: int = 5,
    features: tuple[int, ...] = (64, 128, 256, 512, 1024),
    bilinear: bool = True,
    scale_factor: int = 4,
    **kwargs,
) -> UNet2_5DGenerator:
    """Create a 2.5D UNet Generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_slices: Number of slices to stack for 2.5D processing
        features: Feature dimensions for each UNet level
        bilinear: Whether to use bilinear upsampling
        scale_factor: Super-resolution scale factor
        **kwargs: Additional arguments (ignored)

    Returns:
        UNet2_5DGenerator instance

    """
    return UNet2_5DGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_slices=num_slices,
        features=features,
        bilinear=bilinear,
        scale_factor=scale_factor,
    )
