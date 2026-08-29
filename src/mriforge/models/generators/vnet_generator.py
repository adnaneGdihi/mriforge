"""VNet Generator Implementation for 3D Medical Image Super-Resolution.

VNet is a volumetric CNN architecture specifically designed for
3D medical imaging. It uses residual connections and volumetric
convolutions for efficient 3D processing.

Architecture:
- Encoder-Decoder with skip connections
- Volumetric residual blocks
- PReLU activations
- Instance normalization for stable training
"""

import torch
from torch import nn

from mriforge.models.registry import register_model

from .base_3d_generator import Base3DGenerator, standardize_3d_params


class VNetResidualBlock(nn.Module):
    """Residual block for VNet architecture."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.InstanceNorm3d(out_channels)

        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for VNetResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.prelu(out)
        return out


class VNetEncoderBlock(nn.Module):
    """Encoder block with downsampling for VNet."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.residual_block = VNetResidualBlock(in_channels, out_channels)
        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for VNetEncoderBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.residual_block(x)
        x = self.downsample(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class VNetDecoderBlock(nn.Module):
    """Decoder block with upsampling for VNet."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.bn = nn.InstanceNorm3d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.residual_block = VNetResidualBlock(out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            skip (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for VNetDecoderBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            skip (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.upsample(x)
        x = self.bn(x)
        x = self.prelu(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # Adjust channels if needed
            if x.shape[1] != self.residual_block.conv1.in_channels:
                adjust_conv = nn.Conv3d(
                    x.shape[1],
                    self.residual_block.conv1.in_channels,
                    kernel_size=1,
                    bias=False,
                ).to(x.device)
                x = adjust_conv(x)

        x = self.residual_block(x)
        return x


@register_model(name="vnet", training_mode="reconstruction")
class VNetGenerator(Base3DGenerator):
    """VNet Generator for 3D Medical Image Super-Resolution.

    A volumetric CNN architecture optimized for 3D medical imaging tasks.
    Uses residual connections and volumetric operations for
    efficient processing.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        output_activation: nn.Module | None = None,
    ):
        """Initialize VNet Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_channels: Base number of channels for the first level
            num_levels: Number of encoder/decoder levels
            output_activation: Output activation function

        """
        # Convert VNet-specific parameters to standardized features tuple
        features = tuple(base_channels * (2**i) for i in range(num_levels))

        # Standardize parameters
        params = standardize_3d_params(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            output_activation=output_activation,
            depth=16,  # Default spatial dimensions
            height=16,
            width=16,
        )

        super().__init__(**params)
        self.base_channels = base_channels
        self.num_levels = num_levels

        # Initial convolution
        self.initial_conv = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.initial_bn = nn.InstanceNorm3d(base_channels)
        self.initial_prelu = nn.PReLU(base_channels)

        # Encoder blocks
        self.encoders = nn.ModuleList()
        for i in range(num_levels):
            in_ch = base_channels * (2**i)
            out_ch = base_channels * (2 ** (i + 1))
            self.encoders.append(VNetEncoderBlock(in_ch, out_ch))

        # Decoder blocks
        self.decoders = nn.ModuleList()
        for i in range(num_levels):
            in_ch = base_channels * (2 ** (num_levels - i))
            out_ch = base_channels * (2 ** (num_levels - i - 1))
            self.decoders.append(VNetDecoderBlock(in_ch, out_ch))

        # Final convolution
        self.final_conv = nn.Conv3d(
            base_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )
        self.final_activation = nn.Tanh()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization handled by centralized weight_initialization module."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VNet.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Output tensor [B, C, D, H, W]

        """
        # Initial processing
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_prelu(x)

        # Encoder path with skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)
            x = encoder(x)

        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[-(i + 1)] if i < len(skip_connections) else None
            x = decoder(x, skip)

        # Final output
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape.

        Args:
            input_shape: Input shape (B, C, D, H, W)

        Returns:
            Output shape (B, C, D, H, W)

        """
        # VNet maintains spatial dimensions
        return (
            input_shape[0],
            self.out_channels,
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )


def create_vnet_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 4,
    output_activation: nn.Module | None = None,
    **kwargs,
) -> VNetGenerator:
    """Factory function to create VNet generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        num_levels: Number of encoder/decoder levels
        output_activation: Output activation function
        **kwargs: Additional arguments passed to VNetGenerator

    Returns:
        VNetGenerator instance

    """
    return VNetGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        output_activation=output_activation,
    )
