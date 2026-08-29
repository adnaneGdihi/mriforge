"""Restormer (Efficient Transformer for High-Resolution Image Restoration)
Generator
================================================================================

Implementation of the Restormer architecture for image super-resolution.
Restormer uses efficient transformer blocks with multi-head self-attention
for high-quality image restoration.

Reference:
- "Restormer: Efficient Transformer for High-Resolution Image Restoration"
  by Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat,
  Fahad Shahbaz Khan, Ming-Hsuan Yang
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class FeedForward(nn.Module):
    """Feed-forward network with depthwise convolution."""

    def __init__(self, dim: int, ffn_expansion_factor: int = 2.66, bias: bool = False):
        """__init__.

        Args:
            dim (int): Description.
            ffn_expansion_factor (int): Description.
            bias (bool): Description.
        """
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward network.

        forward method for FeedForward.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim: int, num_heads: int = 8, bias: bool = False):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            bias (bool): Description.
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention module.

        forward method for Attention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)

        return out


class RestormerBlock(nn.Module):
    """Transformer block with attention and feed-forward network."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            ffn_expansion_factor (int): Description.
            bias (bool): Description.
            LayerNorm_type (str): Description.
        """
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        forward method for RestormerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LayerNorm(nn.Module):
    """Layer normalization with different implementations."""

    def __init__(self, dim: int, LayerNorm_type: str = "WithBias"):
        """__init__.

        Args:
            dim (int): Description.
            LayerNorm_type (str): Description.
        """
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = nn.InstanceNorm2d(dim, affine=False)
        else:
            self.body = nn.InstanceNorm2d(dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer normalization.

        forward method for LayerNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h, w = x.shape[-2:]
        return self.body(x)


@register_model(name="restormer", training_mode="reconstruction")
class RestormerGenerator(nn.Module, IGenerator):
    """Restormer Generator for image super-resolution.

    This implementation follows the Restormer architecture with efficient
    transformer blocks for high-quality image restoration.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: int = 6,
        num_refinement_blocks: int = 6,
        heads: int = 8,
        ffn_expansion_factor: int = 2.66,
        bias: bool = False,
        scale: int = 4,
    ):
        """Initialize Restormer generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dim: Number of feature channels
            num_blocks: Number of transformer blocks
            num_refinement_blocks: Number of refinement blocks
            heads: Number of attention heads
            ffn_expansion_factor: Feed-forward expansion factor
            bias: Whether to use bias in convolutions
            scale: Upscaling factor

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        # Validate that dim is divisible by heads
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # Encoder
        self.encoder_level1 = nn.Sequential(
            *[
                RestormerBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ],
        )

        # Downsampling
        self.down1_2 = nn.Conv2d(
            dim,
            dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )
        self.encoder_level2 = nn.Sequential(
            *[
                RestormerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ],
        )

        self.down2_3 = nn.Conv2d(
            int(dim * 2),
            dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )
        self.encoder_level3 = nn.Sequential(
            *[
                RestormerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ],
        )

        # Decoder
        self.up3_2 = nn.ConvTranspose2d(
            int(dim * 4),
            int(dim * 2),
            kernel_size=2,
            stride=2,
            padding=0,
            bias=bias,
        )
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 * 2),
            int(dim * 2),
            kernel_size=1,
            bias=bias,
        )
        self.decoder_level2 = nn.Sequential(
            *[
                RestormerBlock(
                    dim=int(dim * 2),
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ],
        )

        self.up2_1 = nn.ConvTranspose2d(
            int(dim * 2),
            dim,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=bias,
        )
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(
            *[
                RestormerBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ],
        )

        # Refinement
        self.refinement = nn.Sequential(
            *[
                RestormerBlock(
                    dim=dim,
                    num_heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                )
                for _ in range(num_refinement_blocks)
            ],
        )

        # Output
        if scale == 4:
            self.output = nn.Sequential(
                nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, bias=bias),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, bias=bias),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias),
            )
        elif scale == 2:
            self.output = nn.Sequential(
                nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1, bias=bias),
                nn.PixelShuffle(2),
                nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias),
            )
        elif scale == 1:
            self.output = nn.Conv2d(
                dim,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        elif scale > 1:
            # Any other integer factor, via ONE PixelShuffle(scale). The
            # cascaded x2 forms above are kept verbatim so no existing arm's
            # parameter count or weight layout changes; this branch only adds
            # factors they could not express. It exists because the physics
            # picks the factor, not the backbone: the measured ULF-to-3T T1w
            # in-plane gap is 3.26x, and rounding it to 2 or 4 to suit a
            # power-of-two tail would train one decimation and report against
            # another (see SubvoxelRegistrationConfig's sr_scale guard).
            self.output = nn.Sequential(
                nn.Conv2d(dim, dim * scale * scale, kernel_size=3, padding=1, bias=bias),
                nn.PixelShuffle(scale),
                nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=bias),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}. Must be a positive integer.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Restormer generator.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Output tensor of shape [N, out_channels, H*scale, W*scale]

        forward method for RestormerGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Patch embedding
        inp_enc_level1 = self.patch_embed(x)

        # Encoder
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Decoder
        out_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = torch.cat([out_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        out_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([out_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Refinement
        out_dec_level1 = self.refinement(out_dec_level1)

        # Output
        out = self.output(out_dec_level1)

        return out

    @property
    def name(self) -> str:
        """Return the model name."""
        return "RestormerGenerator"

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for Restormer, z is the input image."""
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


def create_restormer_generator(
    in_channels: int = 3,
    out_channels: int = 3,
    dim: int = 48,
    num_blocks: int = 6,
    num_refinement_blocks: int = 6,
    heads: int = 8,
    scale: int = 4,
) -> RestormerGenerator:
    """Factory function to create Restormer generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dim: Number of feature channels
        num_blocks: Number of transformer blocks per level
        num_refinement_blocks: Number of refinement blocks
        heads: Number of attention heads
        scale: Upscaling factor

    Returns:
        RestormerGenerator instance

    """
    return RestormerGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        scale=scale,
    )
