"""Swin Transformer Generator
============================

A standard Swin Transformer architecture adapted for image generation/reconstruction tasks.
This is the STANDARD implementation using linear layers (NOT KAN-based).

For the KAN-based variant, see `swin_transformer_kan.py`.
"""

import torch
from torch import nn

from mriforge.models.blocks.swin import SwinBlock
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(name="transformer_unet", training_mode="reconstruction")
class SwinTransformerGenerator(nn.Module, IGenerator):
    """Swin Transformer Generator for Image Reconstruction.

    This is the STANDARD Swin Transformer implementation using linear layers.
    For the KAN-based variant, use SwinTransformerKANGenerator from swin_transformer_kan.py.

    Key Differences from KAN variant:
    - Uses standard nn.Linear for FFN layers (not KAN)
    - Uses standard nn.LayerNorm
    - No spline-based function approximation
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        **kwargs,
    ):
        """Initialize Swin Transformer Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            img_size: Input image size (assumes square)
            patch_size: Patch size for embedding
            embed_dim: Embedding dimension
            depths: Number of blocks at each stage
            num_heads: Number of attention heads at each stage
            window_size: Window size for local attention
            mlp_ratio: MLP expansion ratio
            qkv_bias: Add bias to QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.window_size = window_size

        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Build stages (using first stage config for simplicity in U-Net like structure)
        self.stages = nn.ModuleList()
        current_res = self.patches_resolution
        for i_stage, (depth, heads) in enumerate(zip(depths, num_heads, strict=False)):
            stage = nn.ModuleList(
                [
                    SwinBlock(
                        dim=embed_dim,
                        num_heads=heads,
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        dropout=drop_rate,
                        attn_dropout=attn_drop_rate,
                        input_resolution=current_res,  # Pass resolution for mask init
                    )
                    for j in range(depth)
                ]
            )
            self.stages.append(stage)

        # Final normalization and projection
        self.final_norm = nn.LayerNorm(embed_dim)

        # Decoder: project back to patch space then reshape
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Reconstructed tensor [B, C_out, H, W]

        forward method for SwinTransformerGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.patch_norm(x)
        x = self.pos_drop(x)

        # Apply Swin stages
        for stage in self.stages:
            for blk in stage:
                x = blk(x)

        # Final normalization
        x = self.final_norm(x)

        # Decode back to image
        x = self.decoder(x)  # [B, num_patches, P*P*out_channels]

        # Reshape to image
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        x = x.view(
            B,
            num_patches_h,
            num_patches_w,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H, W)

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate samples (alias for forward)."""
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """Model name."""
        return "swin_transformer"
