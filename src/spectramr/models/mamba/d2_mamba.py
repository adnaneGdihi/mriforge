"""Dual-Domain Spectral Mamba (D2-Mamba) for MRI Reconstruction.

Innovation IV from SOTA Roadmap: Energy-aware k-space/image processing with linear complexity.

Key Insight:
    Standard Mamba scans images row-by-row or with fixed patterns. For MRI:
    - k-space has energy concentrated at center (low frequencies)
    - Images benefit from hierarchical spatial patterns

    D2-Mamba uses domain-specific scanning:
    - k-space branch: Concentric circles from center (energy-aware)
    - Image branch: Hilbert/Z-order curves (locality preserving)

Mathematical Foundation:
    SSM: h'(t) = Ah(t) + Bx(t)
         y(t) = Ch(t) + Dx(t)

    Discretized: h_k = Āh_{k-1} + B̄x_k

Benefits:
    - O(L) complexity vs O(L²) for Transformers
    - Domain-aware scanning preserves MRI physics
    - Cross-domain gating for k-space/image fusion

References:
    - [15] arxiv:2501.08163 - D2-Mamba for medical imaging
    - [16] S4/Mamba for structured state spaces

Author: Physics-AI MRI Framework
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.registry import register_model
from spectramr.models.utils.shape_validator import assert_shape_equals, validate_tensor_shape


class D2MambaBlock(nn.Module):
    """Dual-Domain Mamba block with k-space and image branches.

    Processes input in both domains with domain-specific scanning,
    then fuses features via gated cross-attention.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """__init__.

        Args:
            d_model (int): Description.
            d_state (int): Description.
            d_conv (int): Description.
            expand (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.d_model = d_model

        # k-space branch (concentric scanning) - Bidirectional
        self.kspace_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.kspace_mamba_rev = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Image branch (Hilbert/Z-order scanning) - Bidirectional
        self.image_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.image_mamba_rev = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Cross-domain gating
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

        # PERF (2026-07-01): cached (indices, inverse) per scan kind, shape
        # and device. ``_get_zorder_indices`` is a pure-Python H×W loop with a
        # 16-iteration bit-interleave per pixel (~2M Python ops at 256²) and
        # was re-run TWICE per forward (scan + unscan), plus a fresh
        # ``argsort`` per unscan; ``_get_concentric_indices`` had the same
        # rebuild-per-call pattern. Nothing in either builder is
        # data-dependent, so build once per (kind, H, W, device).
        self._scan_index_cache: dict[
            tuple[str, int, int, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def forward(
        self,
        x: torch.Tensor,
        domain: Literal["kspace", "image", "dual"] = "dual",
    ) -> torch.Tensor:
        """Forward pass with domain-specific processing.

        Args:
            x: Input tensor (B, C, H, W)
            domain: Which domain(s) to process

        Returns:
            Processed tensor (B, C, H, W)
        """
        B, C, H, W = x.shape

        if domain == "kspace":
            # Only k-space processing
            seq = self._kspace_scan(x)
            out_fwd = self.kspace_mamba(seq)
            out_rev = self.kspace_mamba_rev(seq.flip([1])).flip([1])
            out = self._kspace_unscan(out_fwd + out_rev, H, W)

        elif domain == "image":
            # Only image processing
            seq = self._image_scan(x)
            out_fwd = self.image_mamba(seq)
            out_rev = self.image_mamba_rev(seq.flip([1])).flip([1])
            out = self._image_unscan(out_fwd + out_rev, H, W)

        elif domain == "dual":
            # Both domains with cross-gating
            kspace_seq = self._kspace_scan(x)
            image_seq = self._image_scan(x)

            # Bidirectional K-Space
            k_fwd = self.kspace_mamba(kspace_seq)
            k_rev = self.kspace_mamba_rev(kspace_seq.flip([1])).flip([1])
            kspace_out = k_fwd + k_rev

            # Bidirectional Image
            i_fwd = self.image_mamba(image_seq)
            i_rev = self.image_mamba_rev(image_seq.flip([1])).flip([1])
            image_out = i_fwd + i_rev

            # Compute gate from concatenated features
            gate_input = torch.cat([kspace_out, image_out], dim=-1)
            gate = self.gate(gate_input)

            # Gated fusion
            fused = gate * kspace_out + (1 - gate) * image_out

            # Final projection
            # Difference feature for complementary info
            combined = torch.cat([fused, image_out - kspace_out], dim=-1)
            out_seq = self.out_proj(combined)
            out_seq = self.norm(out_seq)

            out = self._image_unscan(out_seq, H, W)

        else:
            raise ValueError(
                f"D2MambaBlock.forward: unknown domain {domain!r}; "
                "expected one of 'kspace', 'image', 'dual'"
            )

        return out

    def _kspace_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Scan k-space in concentric circles from center.

        This captures low-to-high frequency progression.
        """
        B, C, H, W = x.shape

        # Generate concentric circle indices
        indices, _ = self._get_scan_indices("concentric", H, W, x.device)

        # Flatten and reorder
        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        x_scanned = torch.gather(x_flat, 2, indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1))

        return x_scanned.transpose(1, 2)  # (B, L, C)

    def _kspace_unscan(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reverse concentric circle scanning."""
        B, L, C = x.shape

        _, inverse_indices = self._get_scan_indices("concentric", H, W, x.device)

        x_flat = x.transpose(1, 2)  # (B, C, L)
        x_unscanned = torch.gather(
            x_flat, 2, inverse_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        )

        return x_unscanned.view(B, C, H, W)

    def _image_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Scan image using Z-order (Morton) curve.

        Z-order preserves 2D locality better than raster scan.
        """
        B, C, H, W = x.shape

        # Use Z-order curve for locality
        indices, _ = self._get_scan_indices("zorder", H, W, x.device)

        x_flat = x.view(B, C, -1)
        x_scanned = torch.gather(x_flat, 2, indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1))

        return x_scanned.transpose(1, 2)

    def _image_unscan(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reverse Z-order scanning."""
        B, L, C = x.shape

        _, inverse_indices = self._get_scan_indices("zorder", H, W, x.device)

        x_flat = x.transpose(1, 2)
        x_unscanned = torch.gather(
            x_flat, 2, inverse_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        )

        return x_unscanned.view(B, C, H, W)

    def _get_scan_indices(
        self, kind: str, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(indices, inverse_indices)`` for a scan ordering.

        The static builders below stay as the single source of the orderings;
        this seam only memoizes their (pure, shape-determined) output plus the
        ``argsort`` inverse so the hot forward path never re-runs the Python
        Morton build or a fresh sort.
        """
        key = (kind, H, W, device)
        cached = self._scan_index_cache.get(key)
        if cached is None:
            if kind == "zorder":
                indices = self._get_zorder_indices(H, W, device)
            elif kind == "concentric":
                indices = self._get_concentric_indices(H, W, device)
            else:  # pragma: no cover - internal misuse guard (pitfall #9)
                raise ValueError(f"Unknown scan kind '{kind}'")
            cached = (indices, torch.argsort(indices))
            self._scan_index_cache[key] = cached
        return cached

    @staticmethod
    def _get_concentric_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate indices for concentric circle scanning from center."""
        cy, cx = H // 2, W // 2

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        # Distance from center
        dist = torch.sqrt((y - cy).float() ** 2 + (x - cx).float() ** 2)

        # Angle for tie-breaking within same radius
        angle = torch.atan2((y - cy).float(), (x - cx).float())

        # Sort by distance, then angle
        sort_key = dist * 1000 + (angle + torch.pi)  # Scale to ensure dist dominates
        indices = torch.argsort(sort_key.flatten())

        return indices

    @staticmethod
    def _get_zorder_indices(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate Z-order (Morton) curve indices."""

        def interleave_bits(x: int, y: int) -> int:
            """Interleave bits of x and y for Morton code."""
            z = 0
            for i in range(16):  # Support up to 65536 x 65536
                z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
            return z

        # Generate Morton codes for each position
        morton_codes = []
        for i in range(H):
            for j in range(W):
                morton_codes.append(interleave_bits(j, i))

        morton_tensor = torch.tensor(morton_codes, device=device)
        indices = torch.argsort(morton_tensor)

        return indices


@register_model(name="d2_mamba", training_mode="reconstruction", spatial_dims=(2,))
class D2Mamba(nn.Module):
    """Full D2-Mamba model for MRI reconstruction.

    Encoder-decoder architecture with D2MambaBlocks.

    Args:
        in_channels: Input channels (2 for complex as real/imag)
        base_dim: Base feature dimension
        depths: Number of blocks at each stage
        d_state: SSM state dimension
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_dim: int = 64,
        depths: tuple[int, ...] = (2, 2, 4, 2),
        d_state: int = 16,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_dim (int): Description.
            depths (tuple[int, ...]): Description.
            d_state (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_dim
        for i, depth in enumerate(depths[:-1]):
            stage = nn.ModuleList([D2MambaBlock(ch, d_state=d_state) for _ in range(depth)])
            self.encoders.append(stage)
            self.downsamples.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.ModuleList(
            [D2MambaBlock(ch, d_state=d_state) for _ in range(depths[-1])]
        )

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoder_projs = nn.ModuleList()  # Project ch*2 -> ch after skip concat + blocks

        for i in range(len(depths) - 2, -1, -1):
            self.upsamples.append(nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1))
            ch //= 2
            stage = nn.ModuleList([D2MambaBlock(ch * 2, d_state=d_state) for _ in range(depths[i])])
            self.decoders.append(stage)
            # Project concatenated ch*2 back to ch for next upsample
            self.decoder_projs.append(nn.Conv2d(ch * 2, ch, 1))

        # Skip connection projection - project before concatenation
        self.skip_projs = nn.ModuleList()
        ch_for_proj = base_dim
        for i in range(len(depths) - 1):
            # Each skip has base_dim * (2**i) channels, project to same as decoder
            self.skip_projs.append(nn.Conv2d(ch_for_proj, ch_for_proj, 1))
            ch_for_proj *= 2

        # Output: after last decoder_proj, x has base_dim channels
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (B, in_channels, H, W)

        Returns:
            Output (B, in_channels, H, W)
        """
        # ✅ Validate input shape
        validate_tensor_shape(x, expected_dims=4, name="D2Mamba input")
        expected_shape = (x.shape[0], self.in_channels, x.shape[2], x.shape[3])
        original_shape = x.shape

        x = self.in_conv(x)

        # Encoder
        skips = []
        for stage, down in zip(self.encoders, self.downsamples, strict=False):
            for block in stage:
                x = x + block(x, domain="dual")
            skips.append(x)
            x = down(x)

        # Bottleneck
        for block in self.bottleneck:
            x = x + block(x, domain="dual")

        # Decoder with corrected skip connection handling
        for i, (up, stage, proj) in enumerate(
            zip(self.upsamples, self.decoders, self.decoder_projs, strict=False)
        ):
            x = up(x)
            skip = skips[-(i + 1)]

            # ✅ Project skip BEFORE concatenation
            skip = self.skip_projs[-(i + 1)](skip)

            # ✅ Concatenate: upsample output + skip
            x = torch.cat([x, skip], dim=1)

            for block in stage:
                x = x + block(x, domain="dual")

            # ✅ Project ch*2 -> ch for next level
            x = proj(x)

        output = self.out_conv(x)

        # ✅ Validate output shape
        expected_out_shape = (
            original_shape[0],
            self.out_channels,
            original_shape[2],
            original_shape[3],
        )
        assert_shape_equals(output, expected_out_shape, name="D2Mamba output")

        return output


def create_d2_mamba(
    in_channels: int = 2,
    base_dim: int = 64,
    d_state: int = 16,
) -> D2Mamba:
    """Factory function for D2-Mamba.

    Args:
        in_channels: Input channels
        base_dim: Base feature dimension
        d_state: SSM state dimension

    Returns:
        Configured D2-Mamba model
    """
    return D2Mamba(
        in_channels=in_channels,
        base_dim=base_dim,
        depths=(2, 2, 4, 2),
        d_state=d_state,
    )


__all__ = [
    "D2Mamba",
    "D2MambaBlock",
    "create_d2_mamba",
]
