"""CycleSR Generator
=================

Implementation of the CycleSR architecture for image super-resolution.
CycleSR uses cycle-consistent training for high-quality image restoration.

Reference:
- "CycleSR: Unsupervised Cycle-consistent Super-Resolution"
  by Xiaoyu Xiang, Yapeng Tian, Yulun Zhang, Yun Fu, Xiao-Ping Zhang
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.blocks import UnifiedResidualBlock
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class UpsampleBlock(nn.Module):
    """Upsampling block for CycleSR."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsample block.

        forward method for UpsampleBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.act(self.conv(x))
        return self.shuffle(x)


@register_model(name="cyclesr", training_mode="gan")
class CycleSRGenerator(nn.Module, IGenerator):
    """CycleSR Generator for image super-resolution.

    This implementation follows the CycleSR architecture with cycle-consistent
    training for high-quality image restoration.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 64,
        n_resblocks: int = 16,
        scale: int = 4,
        bias: bool = False,
    ):
        """Initialize CycleSR generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_feat: Number of feature channels
            n_resblocks: Number of residual blocks
            scale: Upscaling factor
            bias: Whether to use bias in convolutions

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            n_feat,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=bias,
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                UnifiedResidualBlock(n_feat, bias=bias, use_group_norm=False)
                for _ in range(n_resblocks)
            ],
        )

        # Middle convolution
        self.conv2 = nn.Conv2d(
            n_feat,
            n_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # Upsampling blocks
        if scale == 4:
            self.up_blocks = nn.Sequential(
                UpsampleBlock(n_feat, n_feat * 4, bias),
                UpsampleBlock(n_feat, n_feat * 4, bias),
            )
        elif scale == 2:
            self.up_blocks = UpsampleBlock(n_feat, n_feat * 4, bias)
        else:
            raise ValueError(f"Unsupported upsample scale factor: {scale}. Valid choices are: 2, 4")

        # Output convolution
        self.conv3 = nn.Conv2d(
            n_feat,
            out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CycleSR generator.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H*scale, W*scale]

        forward method for CycleSRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() != 4:
            raise ValueError(
                "CycleSRGenerator expects 4D input tensor [N, C, H, W], got "
                + str(x.dim())
                + "D tensor with shape "
                + str(x.shape)
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "CycleSRGenerator expects input with "
                + str(self.in_channels)
                + " channels, got "
                + str(x.shape[1])
                + " channels"
            )

        # Initial convolution
        x = F.relu(self.conv1(x))

        # Residual blocks
        residual = x
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv2(x)
        x = x + residual

        # Upsampling
        x = self.up_blocks(x)

        # Output
        x = self.conv3(x)

        return x

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "CycleSRGenerator"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space."""
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


def create_cyclesr_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    n_feat: int = 64,
    n_resblocks: int = 16,
    scale: int = 4,
) -> CycleSRGenerator:
    """Factory function to create CycleSR generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_feat: Number of feature channels
        n_resblocks: Number of residual blocks
        scale: Upscaling factor

    Returns:
        CycleSRGenerator instance

    """
    return CycleSRGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        n_feat=n_feat,
        n_resblocks=n_resblocks,
        scale=scale,
    )
