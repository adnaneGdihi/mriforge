"""Morton Z-Order Locality Stem for Vision State-Space Models.

Solves the **Markovian Dilution** problem in Vision SSMs (Mamba):
raster-scan flattening separates vertically-adjacent pixels by W steps
in the 1D sequence, destroying 2D spatial coherence.

Morton (Z-order) codes interleave the binary coordinates of a 2D grid,
preserving spatial locality (Quadtree ordering) with O(1) GPU-native
bitwise operations. Combined with patch partitioning, this reduces
sequence length while keeping anatomically neighbouring regions adjacent.

Reference:
    Morton, G.M. (1966). "A computer oriented geodetic data base and
    a new technique in file sequencing." IBM.

    Zhu et al. (2024). "Vision Mamba: Efficient Visual Representation
    Learning with Bidirectional State Space Model." arXiv:2401.09417.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bit-interleaving helpers (GPU-native, O(1) parallel)
# ---------------------------------------------------------------------------


def _expand_bits_2d(v: torch.Tensor) -> torch.Tensor:
    """Expand a 16-bit integer into 32 bits by inserting zero-bits.

    Used to interleave (x, y) coordinates into a single Morton code.

    Args:
        v: Integer tensor of grid coordinates.

    Returns:
        Bit-expanded tensor ready for interleaving.
    """
    v = v.long()
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def _expand_bits_3d(v: torch.Tensor) -> torch.Tensor:
    """Expand a 10-bit integer into 30 bits for 3D Morton codes.

    Args:
        v: Integer tensor of grid coordinates (max 1024).

    Returns:
        Bit-expanded tensor for 3D Z-order interleaving.
    """
    v = v.long()
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def compute_morton_2d(grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the 2D Morton (Z-order) permutation index.

    Args:
        grid_h: Number of rows in the patch grid.
        grid_w: Number of columns in the patch grid.

    Returns:
        Tuple of (forward_permutation, inverse_permutation) tensors.
        Forward maps raster → Z-order; inverse maps Z-order → raster.
    """
    Y, X = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    z_codes = _expand_bits_2d(X.flatten()) | (_expand_bits_2d(Y.flatten()) << 1)
    forward = torch.argsort(z_codes)
    inverse = torch.argsort(forward)
    return forward, inverse


def compute_morton_3d(grid_d: int, grid_h: int, grid_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the 3D Morton (Z-order) permutation index.

    Args:
        grid_d: Depth of the patch grid.
        grid_h: Height of the patch grid.
        grid_w: Width of the patch grid.

    Returns:
        Tuple of (forward_permutation, inverse_permutation) tensors.
    """
    Z, Y, X = torch.meshgrid(
        torch.arange(grid_d),
        torch.arange(grid_h),
        torch.arange(grid_w),
        indexing="ij",
    )
    z_codes = (
        _expand_bits_3d(X.flatten())
        | (_expand_bits_3d(Y.flatten()) << 1)
        | (_expand_bits_3d(Z.flatten()) << 2)
    )
    forward = torch.argsort(z_codes)
    inverse = torch.argsort(forward)
    return forward, inverse


# ---------------------------------------------------------------------------
# Morton Locality Stem
# ---------------------------------------------------------------------------


class MortonLocalityStem(nn.Module):
    """GPU-native Z-order locality-preserving tokenizer for Vision SSMs.

    Combines Conv2D patch partitioning (reduces sequence length) with
    Morton Z-order permutation (preserves 2D spatial locality). The
    resulting 1D sequence groups spatially neighbouring patches together,
    enabling Mamba's causal ODE to maintain coherent 2D context.

    Example:
        For a 256×256 image with patch_size=8:
        - Sequence length: 256/8 × 256/8 = 1024 tokens (not 65,536 pixels)
        - Locality: quadrant-grouped via Z-order curve

    Args:
        in_channels: Number of input channels.
        embed_dim: Embedding dimension for patch tokens.
        patch_size: Spatial extent of each patch.
        image_size: Expected input spatial dimensions (H, W).
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 256,
        patch_size: int = 8,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size

        # Patch embedding: compress micro-texture into semantic macro-tokens
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Pre-compute Morton permutation (shared across all batches)
        fwd, inv = compute_morton_2d(self.grid_h, self.grid_w)
        self.register_buffer("morton_fwd", fwd)
        self.register_buffer("morton_inv", inv)

        # Also pre-compute transposed Morton for quad-scan
        fwd_t, inv_t = compute_morton_2d(self.grid_w, self.grid_h)
        self.register_buffer("morton_fwd_t", fwd_t)
        self.register_buffer("morton_inv_t", inv_t)

        logger.info(
            "[MortonLocalityStem] Initialized: in_ch=%d, embed=%d, "
            "patch=%d, grid=(%d,%d), seq_len=%d",
            in_channels,
            embed_dim,
            patch_size,
            self.grid_h,
            self.grid_w,
            self.grid_h * self.grid_w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize and apply Z-order permutation.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Z-ordered token sequence [B, L, D] where L = grid_h * grid_w.

        forward method for MortonLocalityStem.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        patches = self.proj(x)  # [B, D, grid_h, grid_w]
        seq = rearrange(patches, "b c h w -> b (h w) c")
        return seq[:, self.morton_fwd, :]

    def forward_transposed(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize and apply transposed Z-order permutation.

        Args:
            x: Input image [B, C, H, W].

        Returns:
            Transposed Z-ordered token sequence [B, L, D].
        """
        patches = self.proj(x)  # [B, D, grid_h, grid_w]
        # Transpose the grid before flattening
        seq = rearrange(patches, "b c h w -> b (w h) c")
        return seq[:, self.morton_fwd_t, :]

    def reverse_topology(self, seq: torch.Tensor) -> torch.Tensor:
        """Invert Z-order to restore 2D spatial layout.

        Args:
            seq: Z-ordered sequence [B, L, D].

        Returns:
            2D feature map [B, D, grid_h, grid_w].
        """
        original = seq[:, self.morton_inv, :]
        return rearrange(original, "b (h w) c -> b c h w", h=self.grid_h, w=self.grid_w)


# ---------------------------------------------------------------------------
# Quad-Directional Cross-Scan Fusion (V-Mamba style)
# ---------------------------------------------------------------------------


class QuadScanFusion(nn.Module):
    """4-directional state-space fusion to close topological cuts.

    A single Z-order curve creates directional bias at quadrant boundaries.
    Processing the sequence in 4 directions (forward, reverse, transposed
    forward, transposed reverse) mathematically closes all topological loops,
    ensuring global context regardless of curve origin.

    This is computationally 4× the single-direction cost, but eliminates
    anisotropic shading artifacts common in unidirectional Vision SSMs.

    Reference:
        Liu et al. (2024). "VMamba: Visual State Space Model."
        arXiv:2401.10166.

    Args:
        d_model: Model dimension for the Mamba blocks.
        mamba_kwargs: Additional kwargs passed to each MambaBlock.
    """

    def __init__(self, d_model: int, **mamba_kwargs: object) -> None:
        super().__init__()
        from mriforge.models.blocks.mamba_block import MambaBlock

        self.scan_fwd = MambaBlock(d_model=d_model, **mamba_kwargs)
        self.scan_rev = MambaBlock(d_model=d_model, **mamba_kwargs)
        self.scan_tfwd = MambaBlock(d_model=d_model, **mamba_kwargs)
        self.scan_trev = MambaBlock(d_model=d_model, **mamba_kwargs)

        # Learnable merge weights
        self.merge_weight = nn.Parameter(torch.ones(4) * 0.25)
        self.norm = nn.LayerNorm(d_model)

        logger.info("[QuadScanFusion] Initialized: d_model=%d, 4-directional", d_model)

    def forward(
        self,
        z_fwd: torch.Tensor,
        z_rev: torch.Tensor | None = None,
        z_tfwd: torch.Tensor | None = None,
        z_trev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse 4 scan directions.

        Args:
            z_fwd: Forward Z-ordered sequence [B, L, D].
            z_rev: Reverse sequence (auto-computed if None).
            z_tfwd: Transposed forward sequence (uses z_fwd if None).
            z_trev: Transposed reverse sequence (auto-computed if None).

        Returns:
            Fused output [B, L, D] in forward Z-order space.

        forward method for QuadScanFusion.

        Executes PyTorch tensor operations.

        Args:
            z_fwd (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_rev (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_tfwd (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z_trev (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if z_rev is None:
            z_rev = torch.flip(z_fwd, dims=[1])
        if z_tfwd is None:
            z_tfwd = z_fwd  # Fallback: duplicate if no transposed provided
        if z_trev is None:
            z_trev = torch.flip(z_tfwd, dims=[1])

        o1 = self.scan_fwd(z_fwd)
        o2 = torch.flip(self.scan_rev(z_rev), dims=[1])
        o3 = self.scan_tfwd(z_tfwd)
        o4 = torch.flip(self.scan_trev(z_trev), dims=[1])

        # Weighted sum with softmax-normalized merge weights
        w = torch.softmax(self.merge_weight, dim=0)
        fused = w[0] * o1 + w[1] * o2 + w[2] * o3 + w[3] * o4

        return self.norm(fused)
