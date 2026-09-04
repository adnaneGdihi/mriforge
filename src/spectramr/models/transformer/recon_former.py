"""ReconFormer: Recurrent Pyramid Transformer for MRI.

Based on Cluster.md Section 4.2.
Concept: Apply Vision Transformer layers directly to k-space or image patches.
Here we implement a Swin-based block adaptable for k-space processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spectramr.models.registry import register_model


class ReconSwinBlock(nn.Module):
    """Swin Transformer Block with Windowed Self-Attention.

    Implements window-based multi-head self-attention (W-MSA) as described
    in "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
    """

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0, shift_size=0):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            mlp_ratio (Any): Description.
            shift_size (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: Float[Tensor, "B L D"]) -> Float[Tensor, "B L D"]:
        """Apply windowed self-attention.

        Args:
            x: Input tensor [Batch, Length, Dim] where Length = H*W

        forward method for ReconSwinBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B L D']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, D = x.shape
        H = W = int(L**0.5)  # Assume square spatial dimensions

        shortcut = x
        x = self.norm1(x)

        # Reshape to spatial: [B, H, W, D]
        x = x.view(B, H, W, D)

        # Cyclic shift for shifted window attention
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition into windows: [B*nW, window_size*window_size, D]
        x_windows = self._window_partition(x, self.window_size)

        # Window attention
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)

        # Merge windows back: [B, H, W, D]
        x = self._window_reverse(attn_out, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Flatten back to sequence: [B, L, D]
        x = x.view(B, L, D)
        x = x + shortcut

        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut

        return x

    def _window_partition(self, x: Tensor, window_size: int) -> Tensor:
        """Partition input into non-overlapping windows."""
        B, H, W, C = x.shape

        # Pad if needed
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        # Reshape: [B, H//ws, ws, W//ws, ws, C] -> [B*nW, ws*ws, C]
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        return windows

    def _window_reverse(self, windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
        """Merge windows back to feature map."""
        # Pad dimensions for calculation
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        Hp, Wp = H + pad_h, W + pad_w

        B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
        C = windows.shape[-1]

        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        return x


@register_model(name="recon_former", training_mode="reconstruction")
class ReconFormer(nn.Module):
    """K-Space / Image Domain Transformer for Reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,  # Complex (Real, Imag)
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            embed_dim (int): Description.
            depths (tuple[int, ...]): Description.
            num_heads (tuple[int, ...]): Description.
            window_size (int): Description.
        """
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = nn.Sequential(
                *[
                    ReconSwinBlock(embed_dim * (2**i_layer), num_heads[i_layer], window_size)
                    for _ in range(depths[i_layer])
                ]
            )
            self.layers.append(layer)
            if i_layer < len(depths) - 1:
                # Downsample
                # Simplified merging # IMPL
                self.layers.append(
                    nn.Conv2d(
                        embed_dim * (2**i_layer),
                        embed_dim * (2 ** (i_layer + 1)),
                        kernel_size=2,
                        stride=2,
                    )
                )

        # Refinement / Reconstruction Head
        # Output same resolution as input?
        # Typically U-Net shape is needed for recon.
        # This implementation is just the Encoder part + simple upscale for demo.

        last_dim = embed_dim * (2 ** (len(depths) - 1))
        self.final_upscale = nn.Sequential(
            nn.ConvTranspose2d(last_dim, in_channels, kernel_size=32, stride=32),
            # 4 * 2^(3) = 32 stride total?
            # 4 (patch) * 2 * 2 * 2 = 32.
        )

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        """
        Args:
            x: Input image or k-space (B, 2, H, W)
        """
        # Patch Embed
        x = self.patch_embed(x)  # [B, Embed, H/4, W/4]

        # Flatten for Transformer? Or keep Conv?
        # Standard Swin keeps spatial dims, but we treated Block as accepting [B, L, D] above.
        # Let's adapt.

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        for layer in self.layers:
            if isinstance(layer, nn.Sequential):
                # Swin Blocks
                x_flat = layer(x_flat)
            else:
                # Downsample (Conv2d)
                # Reshape back to image
                L = x_flat.shape[1]
                S = int(L**0.5)
                x_img = x_flat.transpose(1, 2).view(B, -1, S, S)
                x_img = layer(x_img)
                # Flatten again
                x_flat = x_img.flatten(2).transpose(1, 2)

        # Final Upscale
        L = x_flat.shape[1]
        S = int(L**0.5)
        x_img = x_flat.transpose(1, 2).view(B, -1, S, S)

        out = self.final_upscale(x_img)
        return out
