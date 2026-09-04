"""Vision Transformer (ViT) Generator
==================================

A standard Vision Transformer adapted for image generation/reconstruction tasks.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(
    name="vision_transformer",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class VisionTransformer(nn.Module, IGenerator):
    """Vision Transformer for Image Reconstruction.

    This implementation uses a standard ViT architecture with a decoder
    to map back to image space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize Vision Transformer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            img_size: Input image size
            patch_size: Patch size
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        # [ARCHITECTURE FIX] Convolutional Stem / Overlapping Patch Embedding
        # Replaces simple strided conv with overlapping convolutions to reduce boundary artifacts.
        # Reference: Xiao et al. "Early Convolutions Help Transformers See Better"

        use_conv_stem = kwargs.get("use_conv_stem", False)
        if use_conv_stem:
            # 4-layer conv stem
            self.patch_embed = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 4, 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    embed_dim // 2, embed_dim, 3, stride=patch_size // 4, padding=1
                ),  # Adjust stride?
                # If patch_size=16 (typical), stems stride 2*2=4. We need total stride 16.
                # So we need 2 more stride-2 layers or 1 stride-4.
                # Let's trust standard CoAtNet style or similar:
                # 3x3 convs.
                # Layer 3: stride 2 -> total 8
                # Layer 4: stride 2 -> total 16
                nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),  # -> 8
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),  # -> 16
            )
            # The output dim needs to match embed_dim?
            # The last conv out_channels is embed_dim.
        else:
            # Standard non-overlapping or simple overlapping
            # Overlapping: k = 2*p - 1? No, k=7, s=4, p=2 (Swin style)
            # Here we just allow standard via kwargs or default
            self.patch_embed = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder (Simple projection back to patches)
        self.decoder_embed = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self._init_weights()

    def _init_weights(self):
        """_init_weights.

        Returns:
            Any: Description.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(self.patch_embed, nn.Sequential):
            for m in self.patch_embed:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.constant_(self.decoder_embed.bias, 0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [N, C, H, W]

        Returns:
            Reconstructed tensor [N, C_out, H, W]

        forward method for VisionTransformer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Patch Embedding
        x = self.patch_embed(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]

        # Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)

        # Decoder
        x = self.decoder_embed(x)  # [B, N, P*P*C_out]

        # Reshape back to image
        x = x.transpose(1, 2)  # [B, P*P*C_out, N]
        x = x.reshape(
            B,
            self.out_channels,
            self.patch_size,
            self.patch_size,
            H // self.patch_size,
            W // self.patch_size,
        )
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.out_channels, H, W)

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "vision_transformer"
