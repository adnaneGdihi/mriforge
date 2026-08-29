"""TRELLIS transformer building blocks.

High-Fidelity 3D Swin Transformer with Rotary Positional Embeddings (RoPE).
Replaces standard ResNets for global context modeling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-exporting legacy for compatibility if needed, though we primarily use Swin now
# trellis_attention or FeedForwardNet might be needed
from .trellis_attention import FeedForwardNet
from .trellis_norm import LayerNorm32

__all__ = [
    "AbsolutePositionEmbedder",
    "RotaryEmbedding",
    "SwinBlock3D",
    "TransformerBlock",  # legacy name preserved for block_registry import
    "TrellisLegacyBlock",  # Legacy
]


class AbsolutePositionEmbedder(nn.Module):
    """
    Standard learnable absolute positional embedding.
    Legacy support for older transformer blocks.
    """

    def __init__(self, dim: int, max_seq_len: int = 5000):
        """__init__.

        Args:
            dim (int): Description.
            max_seq_len (int): Description.
        """
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for AbsolutePositionEmbedder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t = torch.arange(x.shape[1], device=x.device)
        return x + self.emb(t)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        """__init__.

        Args:
            dim (int): Description.
            max_seq_len (int): Description.
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        # Generate frequencies for the sequence length
        # x: [B, H, W, D, C] or [N, L, C]
        # We assume x is [N_win, L, Heads, HeadDim] during attention or we pass L directly
        """forward.

        Args:
            x (torch.Tensor): Description.
            seq_dim (int): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RotaryEmbedding.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            seq_dim (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Specific User requested format: freqs = einsum(n,d->nd) ...
        # We need (cos, sin) for apply_rope
        return freqs


def apply_rotary_pos_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to x.
    x: [Batch, SeqLen, Heads, HeadDim]
    freqs: [SeqLen, HeadDim/2]
    """
    # Reshape x to [..., HeadDim/2, 2] to rotate
    # But standard implementation:
    # x = [x1, x2]
    # out = [x1*cos - x2*sin, x1*sin + x2*cos]

    # Broadcast freqs: [SeqLen, Dim] -> [1, SeqLen, 1, Dim]
    # freqs contains theta. We need cos(theta), sin(theta)

    freqs = freqs.to(x.device)
    cos = freqs.cos()
    sin = freqs.sin()

    # Expand for broadcasting
    # x shape: [B, L, H, D]
    # freqs shape: [L, D/2]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]
    sin = sin.unsqueeze(0).unsqueeze(2)

    # We need to repeat cos/sin for the pairs
    # Assume x is standard [..., D] where D is even
    x1, x2 = x.chunk(2, dim=-1)

    # Rotate
    # (x1 + iy1) * (cos + isin) = x1c - x2s + i(x1s + x2c)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


class WindowAttention3D(nn.Module):
    """Window based multi-head self attention (W-MSA) with RoPE."""

    def __init__(self, dim, num_heads, window_size=7):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        # x: [B_, L, C]  (B_ = Batch * N_Windows)
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WindowAttention3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, L, C = x.shape

        qkv = self.qkv(x).reshape(B_, L, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, L, H, D]

        # Apply RoPE
        freqs = self.rope(q, seq_dim=1)  # [L, D/2]
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        # Attention
        # Scaled Dot Product
        # q: [B, L, H, D] -> [B, H, L, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # PyTorch 2.0 SDPA
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # [B, H, L, D] -> [B, L, H, D] -> [B, L, C]
        x = x.transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)
        return x


class SwinBlock3D(nn.Module):
    """SwinBlock3D class."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=3, mlp_ratio=4.0):
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
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = LayerNorm32(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size)

        self.norm2 = LayerNorm32(dim)
        self.mlp = FeedForwardNet(dim, mlp_ratio=mlp_ratio)

    def cyclic_shift(self, x):
        # x: [B, H, W, D, C]
        """cyclic_shift.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        if self.shift_size > 0:
            return torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                dims=(1, 2, 3),
            )
        return x

    def cyclic_shift_reverse(self, x):
        """cyclic_shift_reverse.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        if self.shift_size > 0:
            return torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3),
            )
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, D, C)
        Returns:
            windows: (num_windows, window_size^3, C)
        """
        B, H, W, D, C = x.shape
        Ws = self.window_size

        # Check alignment
        # In practice, we should pad, but assuming pre-padded or aligned for now

        x = x.view(B, H // Ws, Ws, W // Ws, Ws, D // Ws, Ws, C)
        # Permute to (B, H//Ws, W//Ws, D//Ws, Ws, Ws, Ws, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        # Flatten
        windows = x.view(-1, Ws * Ws * Ws, C)
        return windows

    def window_reverse(self, windows, B, H, W, D):
        """
        Args:
            windows: (num_windows, window_size^3, C)
        Returns:
            x: (B, H, W, D, C)
        """
        Ws = self.window_size
        C = windows.shape[-1]
        x = windows.view(B, H // Ws, W // Ws, D // Ws, Ws, Ws, Ws, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, H, W, D, C)
        return x

    def forward(self, x):
        # x: [B, C, H, W, D] -> Permute to channels last
        # Optimized to keep it Channel First if possible, but Swin is easier Channels Last
        # The user's snippet expected: x = x.permute(0, 2, 3, 4, 1)

        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SwinBlock3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        shortcut = x
        B, C, H, W, D = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, H, W, D, C]

        # Pad if needed? Assuming aligned for this high-fidelity block

        shortcut_last = x.clone()

        # 1. Cyclic Shift
        shifted_x = self.cyclic_shift(x)

        # 2. Window Partition
        x_windows = self.window_partition(shifted_x)  # [N_win, L, C]

        # 3. Attention
        attn_windows = self.attn(x_windows)

        # 4. Reverse Partition
        shifted_x = self.window_reverse(attn_windows, B, H, W, D)

        # 5. Reverse Shift
        x = self.cyclic_shift_reverse(shifted_x)

        # Residual + Norm
        x = shortcut_last + self.norm1(x)

        # MLP
        x = x + self.mlp(self.norm2(x))

        # Back to Channel First
        x = x.permute(0, 4, 1, 2, 3)

        return x


# Include Legacy for compatibility if imports are checking it
class TrellisLegacyBlock(nn.Module):
    """
    Minimal residual transformer block used for staging.
    Legacy: Prefer SwinBlock3D for new architectures.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
        """
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        # Using basic FeedForwardNet as placeholder for Attn in this legacy block
        # (Based on what was seen in the file view, it used FeedForwardNet for 'attn' ??
        #  That was weird in the original file, checking...
        #  Original: self.attn = FeedForwardNet(channels, mlp_ratio=1.0)
        #  Yes, the original 'TransformerBlock' was a dummy block using MLP as attention)
        self.attn = FeedForwardNet(channels, mlp_ratio=1.0)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TrellisLegacyBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm1(x)
        x = x + self.attn(h)
        h = self.norm2(x)
        return x + self.mlp(h)


# Legacy alias preserved so ``block_registry.py`` can import the original
# ``TransformerBlock`` name. Without this, the silent ``except Exception``
# in block_registry.py would set TransformerBlock=None and skip
# registration of ``trellis_transformer_block`` entirely. See
# TODO/audit/10_models_blocks_layers.md F5.
TransformerBlock = TrellisLegacyBlock
