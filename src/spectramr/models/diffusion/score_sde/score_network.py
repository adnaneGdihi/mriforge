"""Score Network Implementation for SDE-based Generative Modeling.

Implements a Time-Conditioned U-Net that predicts the score function
∇_x log p_t(x).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embed_dim: int = 256, scale: float = 30.0):
        """__init__.

        Args:
            embed_dim (int): Description.
            scale (float): Description.
        """
        super().__init__()
        # Randomly sampled weights for Fourier features
        # W ~ N(0, scale^2)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.embed_dim = embed_dim

    def forward(self, x: Float[Tensor, B]) -> Float[Tensor, "B Embed"]:
        # x_proj = x * W * 2 * pi
        """forward.

        Args:
            x (Float[Tensor, B]): Description.
        Returns:
            Float[Tensor, 'B Embed']: Description.

        forward method for GaussianFourierProjection.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B Embed']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Linear layer with activation."""

    def __init__(self, input_dim: int, output_dim: int):
        """__init__.

        Args:
            input_dim (int): Description.
            output_dim (int): Description.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for Dense.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.act(self.dense(x))


class ScoreNetwork(nn.Module):
    """Time-dependent Score Network (NCSN++ style simplified for 3D)."""

    def __init__(
        self,
        in_channels: int = 1,
        nf: int = 32,
        ch_mult: list[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            nf (int): Description.
            ch_mult (list[int]): Description.
            num_res_blocks (int): Description.
        """
        super().__init__()
        self.act = nn.SiLU()
        self.in_channels = in_channels

        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=nf),
            Dense(nf, nf * 4),
            Dense(nf * 4, nf * 4),
        )

        # Downsampling path
        self.conv1 = nn.Conv3d(in_channels, nf, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        curr_ch = nf

        # Using simple ResBlocks for now (would typically use Multi-head attention at lowest res)
        # To avoid massive file complexity, simplified to Conv-based ResBlocks
        # Note: Multi-head attention blocks for lowest resolution are omitted for simplicity.

        # Keeping track of skip connection channels for concat
        self.skip_channels = [nf]

        # Down
        for i, mult in enumerate(ch_mult):
            out_ch = nf * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock(curr_ch, out_ch, time_emb_dim=nf * 4))
                curr_ch = out_ch
                self.skip_channels.append(curr_ch)

            if i != len(ch_mult) - 1:
                self.down_blocks.append(Downsample(curr_ch))
                # Do not append skip for Downsample to maintain N-N symmetry

        # Middle
        self.mid_block1 = ResnetBlock(curr_ch, curr_ch, time_emb_dim=nf * 4)
        self.mid_block2 = ResnetBlock(curr_ch, curr_ch, time_emb_dim=nf * 4)

        # Up
        self.up_blocks = nn.ModuleList()
        # Do not pop mid skip (maintains N-N symmetry)

        # Reverse through mults
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = nf * mult

            for _ in range(num_res_blocks):  # Match number of blocks in down path
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResnetBlock(curr_ch + skip_ch, out_ch, time_emb_dim=nf * 4))
                curr_ch = out_ch

            if i != len(ch_mult) - 1:
                self.up_blocks.append(Upsample(curr_ch))

        self.norm_out = nn.GroupNorm(4, curr_ch)
        self.conv_out = nn.Conv3d(curr_ch, in_channels, 3, padding=1)

    def forward(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> Float[Tensor, "B C D H W"]:
        # Embed time
        # [B, nf*4]
        """forward.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
            t (Float[Tensor, B]): Description.
        Returns:
            Float[Tensor, 'B C D H W']: Description.

        forward method for ScoreNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B C D H W']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        embed = self.embed(t)

        # Initial conv
        h = self.conv1(x)
        skips = [h]

        # Down
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                h = block(h)
                # No skip append for Downsample
            else:
                h = block(h, embed)
                skips.append(h)

        # Middle
        # No pop needed, symmetric skips

        h = self.mid_block1(h, embed)
        h = self.mid_block2(h, embed)

        # Up
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                h = block(h)
            else:
                if len(skips) > 0:
                    skip = skips.pop()
                    # Resize skip if necessary (safety for edge cases, though symmetry should handle it)
                    if skip.shape[2:] != h.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")
                    h = torch.cat([h, skip], dim=1)

                h = block(h, embed)

        h = self.act(self.norm_out(h))
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    """Pre-activation ResNet block with GroupNorm and Time Embedding."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        """__init__.

        Args:
            in_ch (Any): Description.
            out_ch (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        """forward.

        Args:
            x (Any): Description.
            t_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for ResnetBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding (broadcast over spatial dims)
        t_vec = self.time_proj(self.act(t_emb))
        h = h + t_vec[:, :, None, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Downsample class."""

    def __init__(self, channels):
        """__init__.

        Args:
            channels (Any): Description.
        """
        super().__init__()
        self.op = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Downsample.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.op(x)


class Upsample(nn.Module):
    """Upsample class."""

    def __init__(self, channels):
        """__init__.

        Args:
            channels (Any): Description.
        """
        super().__init__()
        self.op = nn.ConvTranspose3d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Upsample.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.op(x)
