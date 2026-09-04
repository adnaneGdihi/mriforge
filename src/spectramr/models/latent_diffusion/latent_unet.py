#!/usr/bin/env python3
"""Latent Diffusion U-Net Module
============================

This module implements a U-Net architecture designed for latent
diffusion models. It operates in latent space with conditioning
capabilities for guided generation.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.normalization import AdaptiveGroupNorm

from ..blocks.embeddings import (
    SinusoidalPositionEmbedding as SinusoidalPositionEmbeddings,
)


class LatentUNet(nn.Module):
    """U-Net architecture for latent diffusion models.

    This model operates in latent space and supports conditioning for
    guided generation. It follows the standard U-Net architecture with
    time conditioning and optional cross-attention for conditioning
    inputs.

    Args:
        in_channels: Number of input channels (latent dimension)
        out_channels: Number of output channels (latent dimension)
        time_emb_dim: Dimension of time embeddings
        cond_channels: Number of conditioning channels (0 for unconditional)
        base_channels: Base number of channels for the first layer
        channel_mult: Multipliers for channel progression
        num_res_blocks: Number of residual blocks per level
        attention_levels: Levels where to apply attention (if using attention)
        dropout: Dropout probability

    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        time_emb_dim: int = 128,
        cond_channels: int = 0,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (2, 3),
        dropout: float = 0.1,
        context_dim: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            cond_channels (int): Description.
            base_channels (int): Description.
            channel_mult (tuple[int, ...]): Description.
            num_res_blocks (int): Description.
            attention_levels (tuple[int, ...]): Description.
            dropout (float): Description.
            context_dim (Optional[int]): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.cond_channels = cond_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.dropout = dropout
        self.context_dim = context_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # [FIX 12] Physics-Aware Conditioning Network
        # For MRI reconstruction/SR, condition on Low-Field / Undersampled Image
        self.cond_net = None
        self.physics_encoder = None

        if cond_channels > 0:
            # Simple conditioning (for class labels, etc.)
            self.cond_net = nn.Sequential(
                nn.Conv2d(cond_channels, base_channels // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    base_channels // 4,
                    base_channels // 2,
                    kernel_size=3,
                    padding=1,
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(base_channels // 2, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

            # [FIX 12] Physics Encoder for Image Conditioning (ControlNet-style)
            # Maps the physical measurement (Low-Field Image) to latent space
            # for spatial alignment with diffusion process
            self.physics_encoder = nn.Sequential(
                # Initial downsampling to match latent resolution
                nn.Conv2d(
                    cond_channels,
                    base_channels // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(8, base_channels // 2),
                nn.SiLU(),
                nn.Conv2d(
                    base_channels // 2,
                    base_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(16, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1),
            )

        # Encoder
        self.encoder = nn.ModuleList()
        current_channels = in_channels  # Start with input channels

        # Initial convolution - don't expand if already at target channels
        if in_channels == base_channels:
            # Already at base channels, no initial conv needed
            pass
        else:
            # Expand to base channels
            self.encoder.append(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            )
            current_channels = base_channels

        # Downsampling blocks
        for i, mult in enumerate(channel_mult):
            out_channels_enc = base_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.encoder.append(
                    LatentDiffusionResBlock(
                        current_channels,
                        out_channels_enc,
                        time_emb_dim,
                        dropout,
                        use_attention=i in attention_levels,
                        context_dim=context_dim if i in attention_levels else None,
                    ),
                )
                current_channels = out_channels_enc

            # Downsampling (except for last level)
            if i < len(channel_mult) - 1:
                self.encoder.append(
                    nn.Conv2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )

        # Bottleneck (middle block)
        # The instruction implies a specific structure for the middle block,
        # which replaces the single bottleneck ResidualBlock.
        # Assuming 'ch' refers to the current_channels at the bottleneck entry.
        ch_bottleneck = current_channels
        self.middle_block = nn.Sequential(
            LatentDiffusionResBlock(
                ch_bottleneck,
                ch_bottleneck,
                time_emb_dim,
                dropout,
                use_attention=True,
                context_dim=context_dim,
            ),
            SelfAttention(ch_bottleneck),  # Assuming AttentionBlock is SelfAttention
            LatentDiffusionResBlock(
                ch_bottleneck,
                ch_bottleneck,
                time_emb_dim,
                dropout,
                use_attention=True,
                context_dim=context_dim,
            ),
        )
        # Update current_channels after bottleneck if it changes, but here it's ch_bottleneck -> ch_bottleneck

        # Decoder
        self.decoder = nn.ModuleList()
        self.channel_mult_rev = list(reversed(channel_mult))

        # Upsampling blocks
        for i, mult in enumerate(self.channel_mult_rev):
            out_channels_dec = base_channels * mult

            # Calculate input channels after skip connection
            # The skip connection comes from the encoder level with 'mult' channels.
            # The current 'h' comes from the previous decoder level or bottleneck.
            # For the first decoder level, 'h' comes from the middle_block (ch_bottleneck).
            # The skip connection comes from the last encoder level (ch_bottleneck).
            # So, the input to the first decoder block is ch_bottleneck + ch_bottleneck.
            # For subsequent levels, 'h' is current_channels (from previous decoder block)
            # and skip connection is base_channels * mult.
            if i == 0:
                # First decoder level: bottleneck output + skip connection
                # The bottleneck output is ch_bottleneck. The skip connection is also ch_bottleneck.
                iin_channels_dec = ch_bottleneck + ch_bottleneck
            else:
                # Subsequent levels: previous decoder output + skip connection
                # 'current_channels' here refers to the output of the previous decoder block.
                # The skip connection is base_channels * mult.
                iin_channels_dec = current_channels + base_channels * mult

            # Upsampling
            if i > 0:
                self.decoder.append(
                    nn.ConvTranspose2d(
                        current_channels,
                        current_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                )

            # Residual blocks
            for _ in range(num_res_blocks):
                self.decoder.append(
                    LatentDiffusionResBlock(
                        iin_channels_dec,
                        out_channels_dec,
                        time_emb_dim,
                        dropout,
                        use_attention=i in attention_levels,
                        context_dim=context_dim if i in attention_levels else None,
                    ),
                )
                iin_channels_dec = out_channels_dec  # Update for next block
                current_channels = out_channels_dec

        # Final convolution
        self.final_conv = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        time_context: torch.Tensor | None = None,
        contrast_context: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the latent U-Net.

        Args:
            x: Input latent tensor [B, C, H, W]
            t: Time step tensor [B]
            cond: Conditioning tensor [B, cond_channels, H, W] (optional)
            time_context: Pre-computed pure time/context embedding [B, time_emb_dim] (optional)
            contrast_context: Contrast embedding to inject into decoder [B, time_emb_dim] (optional)
            text_context: Text embeddings for cross-attention [B, Seq, Dim] (optional)
                          - If provided, skips time_mlp(t).
                          - Used for Contrast-Aware Conditioning (Exp 32b).

        Returns:
            Output latent tensor [B, out_channels, H, W]

        forward method for LatentUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            cond (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time_context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            contrast_context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            text_context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Time embedding
        if time_context is not None:
            t_emb = time_context
        else:
            t_emb = self.time_mlp(t)

        # Conditioning embedding (for AdaGN modulation)
        cond_emb = None
        if self.cond_net is not None and cond is not None:
            cond_emb = self.cond_net(cond)

        # Combine embeddings for AdaGN
        emb = t_emb
        if cond_emb is not None:
            emb = emb + cond_emb

        # Decoder-specific embedding combining contrast
        emb_decoder = emb
        if contrast_context is not None:
            emb_decoder = emb + contrast_context

        # [FIX 12] Physics Conditioning via Concatenation (ControlNet-style)
        # Map physical measurement (Low-Field Image) to latent space for spatial alignment
        if self.physics_encoder is not None and cond is not None:
            # Encode conditioning image to latent resolution
            cond_latent = self.physics_encoder(cond)
            # Resize if dimensions don't match (cond may be at different resolution)
            if cond_latent.shape[-2:] != x.shape[-2:]:
                cond_latent = F.interpolate(
                    cond_latent, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            # Concatenate with noisy latent for strong spatial conditioning
            x = (
                x + cond_latent
            )  # Additive (residual style) instead of concat to preserve channel count

        # Encoder
        skip_connections = []

        # Handle initial convolution (may or may not exist)
        encoder_idx = 0
        if len(self.encoder) > 0 and isinstance(self.encoder[0], nn.Conv2d):
            # Initial convolution exists
            h = self.encoder[0](x)
            encoder_idx = 1
        else:
            # No initial convolution, start with input
            h = x

        for i, _mult in enumerate(self.channel_mult):
            # Residual blocks
            for _ in range(self.num_res_blocks):
                # Check if it's a LatentDiffusionResBlock or just a layer
                if isinstance(self.encoder[encoder_idx], LatentDiffusionResBlock):
                    h = self.encoder[encoder_idx](h, emb, context=text_context)
                else:
                    # Should not happen for ResBlocks, but safe guard
                    h = self.encoder[encoder_idx](h)
                encoder_idx += 1

            skip_connections.append(h)

            # Downsampling (except for last level)
            if i < len(self.channel_mult) - 1:
                h = self.encoder[encoder_idx](h)
                encoder_idx += 1

        # Bottleneck
        # Manually forward through sequential bottleneck components to pass context
        h = self.middle_block[0](h, emb_decoder, context=text_context)
        h = self.middle_block[1](h)
        h = self.middle_block[2](h, emb_decoder, context=text_context)

        # Decoder
        skip_connections = list(reversed(skip_connections))
        decoder_idx = 0

        for i, _mult in enumerate(self.channel_mult_rev):
            # Upsampling
            if i > 0:
                h = self.decoder[decoder_idx](h)
                decoder_idx += 1

            # Skip connection. Downsampling floors odd sizes while Upsample
            # doubles exactly, so h can land 1px short of its skip (the
            # "Expected size 10 but got size 9" crash on latents whose in-plane
            # size is not a multiple of 2**(len(channel_mult)-1)). Align h to the
            # skip's spatial size so ANY input size is accepted.
            skip = skip_connections[i]
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)

            # Residual blocks
            for _ in range(self.num_res_blocks):
                if isinstance(self.decoder[decoder_idx], LatentDiffusionResBlock):
                    h = self.decoder[decoder_idx](h, emb_decoder, context=text_context)
                else:
                    h = self.decoder[decoder_idx](h)
                decoder_idx += 1

        # Final convolution
        return self.final_conv(h)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        # Assume same spatial dimensions for latent diffusion
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])


# SinusoidalPositionEmbeddings is imported from blocks.embeddings.


class LatentDiffusionResBlock(nn.Module):
    """Residual block with Adaptive Group Normalization (AdaGN).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embeddings
        dropout: Dropout probability
        use_attention: Whether to use attention mechanism

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        use_attention: bool = False,
        context_dim: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
            dropout (float): Description.
            use_attention (bool): Description.
            context_dim (Optional[int]): Description.
        """
        super().__init__()

        self.use_attention = use_attention
        self.use_cross_attention = context_dim is not None and use_attention

        # Main convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Normalization - Use AdaptiveGroupNorm
        # We use a helper to determine groups, imported or defined locally
        # Since get_adaptive_groups is in normalization.py, we imported it.
        # But wait, I need to check if I added the import. I will add it in a separate block or assume I add it.
        # I'll define a local helper map or just use heuristic if I forget import.
        # Better to rely on the import I added/will add.

        # Heuristic for groups if helper not available (just in case)
        def get_groups(c):
            """get_groups.

            Args:
                c (Any): Description.
            Returns:
                Any: Description.
            """
            return min(32, c // 2) if c > 4 else 1

        self.norm1 = AdaptiveGroupNorm(in_channels, get_groups(in_channels), time_emb_dim)
        self.norm2 = AdaptiveGroupNorm(out_channels, get_groups(out_channels), time_emb_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Attention (optional)
        if use_attention:
            self.attention = SelfAttention(out_channels)

        if self.use_cross_attention:
            self.cross_attention = CrossAttention(out_channels, context_dim=context_dim)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor [B, in_channels, H, W]
            emb: Combined time/cond embedding [B, time_emb_dim]
            context: Text embeddings for cross-attention [B, Seq, Dim]

        Returns:
            Output tensor [B, out_channels, H, W]

        forward method for LatentDiffusionResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # First convolution
        h = self.norm1(x, emb)
        h = F.silu(h)
        h = self.conv1(h)

        # Second convolution
        h = self.norm2(h, emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        skip = self.skip_connection(x)

        # Attention (optional)
        if self.use_attention:
            h = self.attention(h)

        if self.use_cross_attention and context is not None:
            h = self.cross_attention(h, context)

        return h + skip


class SelfAttention(nn.Module):
    """Self-attention mechanism for latent features."""

    def __init__(self, channels: int):
        """__init__.

        Args:
            channels (int): Description.
        """
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to input tensor.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W]

        forward method for SelfAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).transpose(1, 2)

        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.transpose(-2, -1).view(
            -1,
            self.channels,
            size[0],
            size[1],
        )


class CrossAttention(nn.Module):
    """Cross-attention mechanism for conditioning on text embeddings."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        """__init__.

        Args:
            query_dim (int): Description.
            context_dim (int): Description.
            heads (int): Description.
            dim_head (int): Description.
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Query features [B, C, H, W]
            context: Context features [B, Tokens, Dim]

        forward method for CrossAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        if context is None:
            return x

        b, c, h, w = x.shape

        # Reshape to [B, Sequence, Dim]
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        # [B, Seq, Heads, DimHead]
        q = q.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)

        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, -1, self.heads * self.dim_head)

        return self.to_out(out).permute(0, 2, 1).contiguous().view(b, c, h, w) + x


# Convenience function for creating latent U-Net
def create_latent_unet(
    in_channels: int = 4,
    out_channels: int = 4,
    time_emb_dim: int = 128,
    cond_channels: int = 0,
    **kwargs,
) -> LatentUNet:
    """Create a latent U-Net for diffusion models.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embeddings
        cond_channels: Number of conditioning channels
        **kwargs: Additional arguments for LatentUNet

    Returns:
        Configured LatentUNet instance

    """
    return LatentUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        time_emb_dim=time_emb_dim,
        cond_channels=cond_channels,
        **kwargs,
    )
