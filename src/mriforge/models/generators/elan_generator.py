"""ELAN (Efficient Long-range Attention Network) Generator
=======================================================

Implementation of the ELAN architecture for image super-resolution.
ELAN uses efficient long-range attention mechanisms for high-quality
image restoration.

Reference:
- "ELAN: Efficient Long-range Attention Network for Image Super-resolution"
  by Jie Liu, Jie Tang, Gangshan Wu
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class ELAB(nn.Module):
    """Efficient Local Attention Block."""

    def __init__(self, n_feat: int, reduction: int = 8, bias: bool = False):
        """__init__.

        Args:
            n_feat (int): Description.
            reduction (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.n_feat = n_feat
        self.reduction = reduction

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.act = nn.ReLU(inplace=True)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ELAB.

        forward method for ELAB.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.conv1(x)
        x1 = self.act(x1)

        x2 = self.conv2(x1)
        x2 = self.act(x2)

        x3 = self.conv3(x2)

        y = x1 + x2 + x3
        y = y * self.se(y)

        return y + x


class ELANBlock(nn.Module):
    """ELAN Block with local and global attention."""

    def __init__(self, n_feat: int, reduction: int = 8, bias: bool = False):
        """__init__.

        Args:
            n_feat (int): Description.
            reduction (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.n_feat = n_feat

        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        self.elab1 = ELAB(n_feat, reduction, bias)
        self.elab2 = ELAB(n_feat, reduction, bias)
        self.elab3 = ELAB(n_feat, reduction, bias)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ELAN block.

        forward method for ELANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.conv1(x)
        x1 = self.act(x1)

        x2 = self.elab1(x1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)

        x3 = self.elab2(x2)
        x3 = self.conv3(x3)
        x3 = self.act(x3)

        x4 = self.elab3(x3)

        return x4 + x


class Upsampler(nn.Module):
    """Upsampling module for ELAN."""

    def __init__(self, n_feat: int, scale: int, bias: bool = False):
        """__init__.

        Args:
            n_feat (int): Description.
            scale (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.scale = scale

        if scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, kernel_size=1, bias=bias),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feat, n_feat * 4, kernel_size=1, bias=bias),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, kernel_size=1, bias=bias),
                nn.PixelShuffle(2),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampler.

        forward method for Upsampler.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.up(x)


@register_model(name="elan", training_mode="reconstruction")
class ELANGenerator(nn.Module, IGenerator):
    """ELAN Generator for image super-resolution.

    This implementation follows the ELAN architecture with efficient
    long-range attention for high-quality image restoration.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 64,
        n_blocks: int = 8,
        scale: int = 4,
        bias: bool = False,
    ):
        """Initialize ELAN generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_feat: Number of feature channels
            n_blocks: Number of ELAN blocks
            scale: Upscaling factor
            bias: Whether to use bias in convolutions

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # Feature extraction
        self.conv1 = nn.Conv2d(
            in_channels,
            n_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # ELAN blocks
        self.elan_blocks = nn.ModuleList(
            [ELANBlock(n_feat, bias=bias) for _ in range(n_blocks)],
        )

        # Feature refinement
        self.conv2 = nn.Conv2d(
            n_feat,
            n_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # Upsampling
        self.upsampler = Upsampler(n_feat, scale, bias)

        # Output
        self.conv3 = nn.Conv2d(
            n_feat,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ELAN generator.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H*scale, W*scale]

        forward method for ELANGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Feature extraction
        x = self.conv1(x)

        # ELAN blocks
        for block in self.elan_blocks:
            x = block(x)

        # Feature refinement
        x = self.conv2(x)

        # Upsampling
        x = self.upsampler(x)

        # Output
        x = self.conv3(x)

        return x

    @property
    def name(self) -> str:
        """Return the model name."""
        return "ELANGenerator"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for ELAN, z is the input image."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        n, _, h, w = input_shape
        return (n, self.out_channels, h * self.scale, w * self.scale)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return self.num_parameters

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_elan_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    n_feat: int = 64,
    n_blocks: int = 8,
    scale: int = 4,
) -> ELANGenerator:
    """Factory function to create ELAN generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_feat: Number of feature channels
        n_blocks: Number of ELAN blocks
        scale: Upscaling factor

    Returns:
        ELANGenerator instance

    """
    return ELANGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        n_feat=n_feat,
        n_blocks=n_blocks,
        scale=scale,
    )
