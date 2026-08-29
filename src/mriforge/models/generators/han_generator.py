"""HAN (Holistic Attention Network) Generator
==========================================

Implementation of the Holistic Attention Network for image super-resolution.
HAN uses a holistic attention mechanism that captures both spatial and
channel-wise dependencies for high-quality image restoration.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class LAM_Module(nn.Module):
    """Local Attention Module for spatial attention."""

    def __init__(self, in_dim: int):
        """__init__.

        Args:
            in_dim (int): Description.
        """
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: input features with shape [N, C, H, W]

        Returns:
            output features with attention applied

        forward method for LAM_Module.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """Channel Attention Module for channel-wise attention."""

    def __init__(self, in_dim: int):
        """__init__.

        Args:
            in_dim (int): Description.
        """
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: input features with shape [N, C, H, W]

        Returns:
            output features with attention applied

        forward method for CAM_Module.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, C, -1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CSAM_Module(nn.Module):
    """Cross-Scale Attention Module combining LAM and CAM."""

    def __init__(self, in_dim: int):
        """__init__.

        Args:
            in_dim (int): Description.
        """
        super().__init__()
        self.lam = LAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both local and channel attention.

        forward method for CSAM_Module.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.lam(x)
        x = self.cam(x)
        return x


class HANBlock(nn.Module):
    """Holistic Attention Network Block."""

    def __init__(
        self,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 16,
        bias: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        """__init__.

        Args:
            n_feat (int): Description.
            kernel_size (int): Description.
            reduction (int): Description.
            bias (bool): Description.
            act (nn.Module): Description.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            n_feat,
            n_feat,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            n_feat,
            n_feat,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        self.conv3 = nn.Conv2d(
            n_feat,
            n_feat,
            kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        self.attention = CSAM_Module(n_feat)

        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HAN block.

        forward method for HANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        res = self.conv1(x)
        res = self.act(res)

        res = self.conv2(res)
        res = self.act(res)

        res = self.conv3(res)

        # Apply attention
        res = self.attention(res)

        return res + x


@register_model(name="han", training_mode="reconstruction")
class HANGenerator(nn.Module, IGenerator):
    """Holistic Attention Network Generator for image super-resolution.

    This implementation follows the HAN architecture with holistic attention
    mechanisms for capturing both spatial and channel dependencies.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_feat: int = 64,
        n_blocks: int = 8,
        scale: int = 4,
    ):
        """Initialize HAN generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_feat: Number of feature channels
            n_blocks: Number of HAN blocks
            scale: Upscaling factor

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # Feature extraction
        self.conv_first = nn.Conv2d(in_channels, n_feat, 3, padding=1, bias=True)

        # HAN blocks
        self.han_blocks = nn.ModuleList([HANBlock(n_feat) for _ in range(n_blocks)])

        # Feature fusion
        self.conv_after_body = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=True)

        # Upsampling
        if scale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
            )
        elif scale == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),
            )
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        # Output
        self.conv_last = nn.Conv2d(n_feat, out_channels, 3, padding=1, bias=True)

        # Skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HAN generator.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H*scale, W*scale]

        forward method for HANGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Skip connection
        skip = self.skip_conv(x)

        # Feature extraction
        feat = self.conv_first(x)

        # HAN blocks
        res = feat
        for block in self.han_blocks:
            res = block(res)

        # Feature fusion
        res = self.conv_after_body(res)
        feat = feat + res

        # Upsampling
        feat = self.upsampler(feat)

        # Output
        out = self.conv_last(feat)

        # Skip connection
        out = out + F.interpolate(
            skip,
            scale_factor=self.get_scale_factor(),
            mode="bicubic",
            align_corners=False,
        )

        return out

    def get_scale_factor(self) -> int:
        """Get the upscaling factor."""
        if hasattr(self, "upsampler"):
            if isinstance(self.upsampler, nn.Identity):
                return 1
            if len(self.upsampler) == 2:  # scale=2
                return 2
            if len(self.upsampler) == 4:  # scale=4
                return 4
        return 1

    @property
    def name(self) -> str:
        """Return the model name."""
        return "HANGenerator"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for HAN, z is the input image."""
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


def create_han_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    n_feat: int = 64,
    n_blocks: int = 8,
    scale: int = 4,
) -> HANGenerator:
    """Factory function to create HAN generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_feat: Number of feature channels
        n_blocks: Number of HAN blocks
        scale: Upscaling factor

    Returns:
        HANGenerator instance

    """
    return HANGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        n_feat=n_feat,
        n_blocks=n_blocks,
        scale=scale,
    )
