"""Swin-Mamba-KAN: Unified Backbone for MRI Reconstruction.

Innovation XV (SOTA 2.0): Grand Unification.

Architecture:
    - Parallel Branch 1: Swin Window Attention (Local)
    - Parallel Branch 2: Mamba SSM (Global)
    - FFN: KANBlock (Functional)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.spectral.fourier_kan import KANBlock

logger = logging.getLogger(__name__)

from spectramr.models.registry import register_model
from spectramr.models.spectral.fourier_kan_swin import (
    WindowAttention,
    window_partition,
    window_reverse,
)


class MambaLayer(nn.Module):
    """Simplified Pure-Torch Mamba (Selective SSM) Layer."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        """__init__.

        Args:
            d_model (Any): Description.
            d_state (Any): Description.
            d_conv (Any): Description.
            expand (Any): Description.
        """
        super().__init__()
        self.d_model = d_model
        # Delegate the selective-SSM to the official mamba_ssm kernel (via
        # MambaBlock) instead of the slow pure-Python for-loop scan this class
        # previously shipped — SwinMambaKAN's novelty is the Swin attention + KAN
        # MLP, not the sequence mixer, so the mixer should be a real SSM.
        self.mamba = MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor):
        # x: [B, L, D]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            Any: Description.

        forward method for MambaLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.mamba(x)


class SwinMambaKANBlock(nn.Module):
    """Hybrid Block: Swin + Mamba + KAN."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)

        # Parallel Branch 1: Swin Attention (Full Dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.window_size = window_size
        self.shift_size = shift_size

        # Parallel Branch 2: Mamba (Full Dim)
        self.mamba = MambaLayer(dim)

        self.norm2 = nn.LayerNorm(dim)
        # FFN: KAN
        self.mlp = KANBlock(dim, int(dim * mlp_ratio))

        # Weights for branches
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor, H: int, W: int):
        # x: [B, L, C] where L = H*W
        """forward.

        Args:
            x (torch.Tensor): Description.
            H (int): Description.
            W (int): Description.
        Returns:
            Any: Description.

        forward method for SwinMambaKANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            H (int): Expected input tensor.
            W (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        x_view = x_norm.view(B, H, W, C)

        # --- Swin Branch ---
        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_view

        # Partition
        x_windows = window_partition(shifted_x, self.window_size)  # [B*nW, Ws*Ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attn
        attn_windows = self.attn(x_windows)  # [B*nW, Ws*Ws, C]

        # Reverse Partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse Shift
        if self.shift_size > 0:
            x_swin = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_swin = shifted_x

        x_swin = x_swin.view(B, L, C)

        # --- Mamba Branch ---
        x_mamba = self.mamba(x_norm)  # [B, L, 16]

        # Combine
        logger.debug(
            f"DEBUG: x_swin={x_swin.shape}, x_mamba={x_mamba.shape}, shortcut={shortcut.shape}"
        )
        x = shortcut + self.alpha * x_swin + self.beta * x_mamba

        # --- KAN FFN ---
        x = x + self.mlp(self.norm2(x))

        return x


@register_model(name="swin_mamba_kan", training_mode="reconstruction")
class SwinMambaKAN(nn.Module):
    """Full Model."""

    def __init__(self, in_chans=2, embed_dim=32, depths=[2, 2], num_heads=[2, 4], window_size=4):
        """__init__.

        Args:
            in_chans (Any): Description.
            embed_dim (Any): Description.
            depths (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
        """
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1), nn.GELU()
        )
        self.pos_drop = nn.Dropout(p=0.0)

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.ModuleList(
                [
                    SwinMambaKANBlock(
                        dim=embed_dim,  # Constant dim for backbon
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                    )
                    for _ in range(depths[i_layer])
                ]
            )
            self.layers.append(layer)

            # Patch Merge layer would go here (Downsample)
            # For reconstruction, we often keep resolution or U-Net.
            # Let's do U-Net like or just flat Deep network for "Backbone" demo.
            # Flat for now, removing downsample to avoid complex shape math in demo.
            # Just changing channels? No, keep constant dim for simplicity or
            # user expects "Backbone" to be usable in UNet.

            # I'll implement a simple isotropic backbone (constant dim).
            # Disregard 2**i_layer expansion for this demo unless I add downsampling.

        self.final = nn.Linear(embed_dim, 2)  # Real/Imag

    def forward(self, x):
        # x: [B, C, H, W]
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SwinMambaKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        x = self.embed(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, Embed]
        x = x.view(B, H * W, -1)

        for i, stage in enumerate(self.layers):
            for blk in stage:
                x = blk(x, H, W)

        x = self.final(x)  # [B, L, 2]
        x = x.view(B, H, W, 2).permute(0, 3, 1, 2)
        return x


def create_swin_mamba_kan() -> SwinMambaKAN:
    """create_swin_mamba_kan.

    Returns:
        SwinMambaKAN: Description.
    """
    return SwinMambaKAN()
