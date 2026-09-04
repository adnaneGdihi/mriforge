import torch
import torch.nn as nn

from spectramr.models.blocks.adapters import TimeFiLM2D
from spectramr.models.blocks.attention import CBAMSpatialAttention, WindowAttention
from spectramr.models.blocks.embeddings import SinusoidalPositionEmbedding


class TimestepFiLMResidualBlock(nn.Module):
    """Residual block with FiLM conditioning for Timestep and Contrast."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        # FiLM layer takes the combined (time + contrast) embedding
        self.film = TimeFiLM2D(out_channels, cond_emb_dim)

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """forward method for TimestepFiLMResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.conv1(x)
        h = self.film(h, cond_emb)
        h = self.conv2(h)
        return h + self.shortcut(x)


class ImageColdDiffusionUNet(nn.Module):
    """
    Image-Space Cold Diffusion U-Net.

    Features:
    - Time embeddings via SinusoidalPositionEmbedding
    - Cross-contrast conditioning via FiLM embeddings (cond)
    - Skip connections (concatenation)
    - Phase-agnostic spatial attention using CBAMSpatialAttention and WindowAttention
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        cond_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels

        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

        # Combined embedding dimension (Time + Cond)
        self.cond_emb_dim = time_dim * 4
        if cond_dim is not None:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim * 4),
            )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 2. Downsampling
        self.downs = nn.ModuleList()
        current_channels = base_channels
        in_ch = base_channels
        self.skip_dims = []

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    TimestepFiLMResidualBlock(in_ch, out_ch, self.cond_emb_dim, dropout)
                )
                in_ch = out_ch

            # Add attention at the lower resolutions
            attn = nn.Identity()
            if i >= len(channel_mults) - 2:
                attn = nn.Sequential(
                    WindowAttention(out_ch, num_heads=min(8, out_ch // 8)),
                    CBAMSpatialAttention(),
                )

            down_layer = (
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                if i != len(channel_mults) - 1
                else nn.Identity()
            )

            self.downs.append(
                nn.ModuleDict({"blocks": level_blocks, "attn": attn, "down": down_layer})
            )
            # Each block produces a skip, plus one for the post-attention output
            self.skip_dims.extend([out_ch] * (num_res_blocks + 1))

        # 3. Bottleneck
        self.bottleneck = nn.ModuleDict(
            {
                "block1": TimestepFiLMResidualBlock(in_ch, in_ch, self.cond_emb_dim, dropout),
                "attn": nn.Sequential(
                    WindowAttention(in_ch, num_heads=min(8, in_ch // 8)),
                    CBAMSpatialAttention(),
                ),
                "block2": TimestepFiLMResidualBlock(in_ch, in_ch, self.cond_emb_dim, dropout),
            }
        )

        # 4. Upsampling
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = self.skip_dims.pop()
                level_blocks.append(
                    TimestepFiLMResidualBlock(in_ch + skip_ch, out_ch, self.cond_emb_dim, dropout)
                )
                in_ch = out_ch

            attn = nn.Identity()
            if i >= len(channel_mults) - 2:
                attn = nn.Sequential(
                    WindowAttention(out_ch, num_heads=min(8, out_ch // 8)),
                    CBAMSpatialAttention(),
                )

            up_layer = (
                nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                if i != 0
                else nn.Identity()
            )

            self.ups.append(nn.ModuleDict({"blocks": level_blocks, "attn": attn, "up": up_layer}))

        # 5. Final Output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy image [B, C, H, W]
            t: Diffusion timesteps [B]
            cond: Cross-contrast or other external embeddings [B, cond_dim]

        forward method for ImageColdDiffusionUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Embeddings
        t_emb = self.time_mlp(t)
        emb = t_emb
        if cond is not None and hasattr(self, "cond_proj"):
            c_emb = self.cond_proj(cond)
            emb = emb + c_emb  # Combine time and contrast embeddings

        # Initial conv
        x = self.init_conv(x)
        skips = []

        # Down
        for level in self.downs:
            for block in level["blocks"]:
                x = block(x, emb)
                skips.append(x)
            x = level["attn"](x)
            skips.append(x)  # Post-attention skip for decoder
            x = level["down"](x)

        # Bottleneck
        x = self.bottleneck["block1"](x, emb)
        x = self.bottleneck["attn"](x)
        x = self.bottleneck["block2"](x, emb)

        # Up
        for level in self.ups:
            for block in level["blocks"]:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)  # Skip connection
                x = block(x, emb)
            x = level["attn"](x)
            x = level["up"](x)

        return self.final_conv(x)
