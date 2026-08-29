r"""Breakthrough 2026 attention-block generators.

Each of the five Batch-4 BREAKTHROUGH attention blocks lives in
:mod:`mriforge.models.blocks` and is mathematically self-contained, but a block
on its own is not reachable from a YAML config — the model registry only
dispatches on classes registered via ``@register_model``. This module
provides five thin generators that drop each block into the bottleneck of
a small U-Net, exposing them to the framework's standard model-by-name
dispatch.

Each generator:

* Accepts a magnitude image of shape ``(B, 1, H, W)`` and returns a
  reconstruction of the same shape.
* Down-samples to a fixed bottleneck (token-grid 32 × 32 by default) so
  the attention block sees a predictable token length.
* Up-samples back to the input spatial size.

The networks are intentionally small (≈ 50 k–150 k parameters) — they are
*proof-of-integration* generators, not production models. Their purpose is
to make each block reachable from a YAML by name. Paired YAMLs live under
``experiments/inprogress/novel_2026/breakthrough_*.yaml``.

For the mathematical background see ``docs/breakthrough_methods`` and the
parent proposal document
``TODO/I'll run the full deep-ideation pipeline.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.bloch_lrs_attention import BlochLRSAttention
from mriforge.models.blocks.lanczos_attention import KrylovLanczosAttention
from mriforge.models.blocks.mps_attention import MatrixProductStateAttention
from mriforge.models.blocks.tjl_hash_attention import TJLHashAttention
from mriforge.models.blocks.toeplitz_attention import ToeplitzSpectralAttention
from mriforge.models.registry import register_model

__all__ = [
    "BlochLRSAttentionUNet",
    "LanczosAttentionUNet",
    "MPSAttentionUNet",
    "TJLMRFEstimator",
    "ToeplitzAttentionUNet",
]

_BOTTLENECK_HW: int = 32
_BOTTLENECK_DIM: int = 32


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
    )


class _AttentionBottleneckUNet(nn.Module):
    """Shared U-Net scaffold; subclasses inject one attention block."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_1 = _conv_block(in_channels, 16)
        self.encoder_2 = _conv_block(16, _BOTTLENECK_DIM)
        self.decoder_2 = _conv_block(_BOTTLENECK_DIM, 16)
        self.decoder_1 = _conv_block(16 + 16, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _to_bottleneck(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Encode to the bottleneck token grid and remember the skip + original size."""
        original_size = x.shape[-2:]
        feat1 = self.encoder_1(x)
        pooled = self.down(feat1)
        feat2 = self.encoder_2(pooled)
        # Resize bottleneck to the fixed (32, 32) token grid expected by the attention block.
        feat2_resized = nn.functional.interpolate(
            feat2,
            size=(_BOTTLENECK_HW, _BOTTLENECK_HW),
            mode="bilinear",
            align_corners=False,
        )
        return feat2_resized, feat1, original_size

    def _from_bottleneck(
        self,
        bottleneck: torch.Tensor,
        skip: torch.Tensor,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        # Resize bottleneck back to the pooled feature grid before the decoder.
        pooled_hw = (original_size[0] // 2, original_size[1] // 2)
        upsampled = nn.functional.interpolate(
            bottleneck, size=pooled_hw, mode="bilinear", align_corners=False
        )
        dec2 = self.decoder_2(upsampled)
        dec2 = self.up(dec2)
        merged = torch.cat([dec2, skip], dim=1)
        dec1 = self.decoder_1(merged)
        return self.head(dec1)

    def _apply_attention(self, bottleneck: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register_model(
    name="toeplitz_attention_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class ToeplitzAttentionUNet(_AttentionBottleneckUNet):
    """U-Net with a Toeplitz-spectral attention bottleneck (P6)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.attn = ToeplitzSpectralAttention(
            embed_dim=_BOTTLENECK_DIM,
            num_heads=4,
            token_len=_BOTTLENECK_HW * _BOTTLENECK_HW,
        )

    def _apply_attention(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        tokens = bottleneck.reshape(b, c, h * w).transpose(1, 2)
        attended = self.attn(tokens)
        return attended.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip, original_size = self._to_bottleneck(x)
        attended = self._apply_attention(bottleneck)
        return self._from_bottleneck(attended, skip, original_size)


@register_model(
    name="bloch_lrs_attention_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class BlochLRSAttentionUNet(_AttentionBottleneckUNet):
    """U-Net with a Bloch-coupled low-rank-plus-sparse attention bottleneck (P7)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        rank: int = 8,
        sparse_k: int = 0,
        bloch_dictionary: torch.Tensor | None = None,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.attn = BlochLRSAttention(
            embed_dim=_BOTTLENECK_DIM,
            token_len=_BOTTLENECK_HW * _BOTTLENECK_HW,
            rank=rank,
            sparse_k=sparse_k,
            bloch_dictionary=bloch_dictionary,
        )

    def _apply_attention(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        tokens = bottleneck.reshape(b, c, h * w).transpose(1, 2)
        attended = self.attn(tokens)
        return attended.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip, original_size = self._to_bottleneck(x)
        attended = self._apply_attention(bottleneck)
        return self._from_bottleneck(attended, skip, original_size)


@register_model(
    name="lanczos_attention_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class LanczosAttentionUNet(_AttentionBottleneckUNet):
    """U-Net with a Krylov-Lanczos attention bottleneck (P8)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        krylov_dim: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.attn = KrylovLanczosAttention(
            embed_dim=_BOTTLENECK_DIM,
            num_heads=num_heads,
            krylov_dim=krylov_dim,
        )

    def _apply_attention(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        tokens = bottleneck.reshape(b, c, h * w).transpose(1, 2)
        attended = self.attn(tokens)
        return attended.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip, original_size = self._to_bottleneck(x)
        attended = self._apply_attention(bottleneck)
        return self._from_bottleneck(attended, skip, original_size)


@register_model(
    name="mps_attention_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class MPSAttentionUNet(_AttentionBottleneckUNet):
    """U-Net with a matrix-product-state attention bottleneck (P10).

    The MPS attention requires ``N = 2^L`` tokens. We use ``L = 10`` for
    the default bottleneck (1024 = 32 × 32 tokens).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bond_dim: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        n_tokens = _BOTTLENECK_HW * _BOTTLENECK_HW
        n_log_tokens = (n_tokens).bit_length() - 1
        if 2**n_log_tokens != n_tokens:
            raise ValueError(
                f"Bottleneck token count {n_tokens} is not a power of 2; "
                "MPS attention requires N = 2^L."
            )
        self.attn = MatrixProductStateAttention(
            embed_dim=_BOTTLENECK_DIM,
            n_log_tokens=n_log_tokens,
            bond_dim=bond_dim,
        )

    def _apply_attention(self, bottleneck: torch.Tensor) -> torch.Tensor:
        b, c, h, w = bottleneck.shape
        tokens = bottleneck.reshape(b, c, h * w).transpose(1, 2)
        attended = self.attn(tokens)
        return attended.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skip, original_size = self._to_bottleneck(x)
        attended = self._apply_attention(bottleneck)
        return self._from_bottleneck(attended, skip, original_size)


@register_model(
    name="tjl_mrf_estimator",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class TJLMRFEstimator(nn.Module):
    r"""Tensorised-JL hashed-attention regressor for MR fingerprinting (P9).

    Unlike the other four breakthrough generators this one is a *regressor*,
    not a reconstructor: it consumes an MRF readout of shape
    ``(B, N, T, C)`` (or, for image-shaped inputs, ``(B, 1, H, W)`` reshaped
    into a single-pixel readout of shape ``(B, H*W, T, C)`` with
    ``T = C = 1``) and returns tissue-parameter estimates of shape
    ``(B, N, out_channels)``.

    For image-shaped inputs compatible with the standard reconstruction
    paradigm, the forward pass auto-reshapes a ``(B, 1, H, W)`` input into
    a single-coil, single-readout MRF signal and returns a
    ``(B, 1, H, W)`` parameter map so the model slots into the same
    audit pipeline as the other four. This keeps the YAML schema simple.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        time_dim: int = 1,
        coil_dim: int = 1,
        hash_dim: int = 8,
        value_dim: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn = TJLHashAttention(
            time_dim=time_dim,
            coil_dim=coil_dim,
            hash_dim=hash_dim,
            value_dim=value_dim,
        )
        self.regress = nn.Linear(value_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == self.in_channels:
            # Image-shaped input: (B, C, H, W) → MRF readout (B, N, 1, 1).
            b, c, h, w = x.shape
            readout = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(-1)
            # Aggregate channels into time axis for the JL hash.
            readout = readout.transpose(2, 3)  # (B, H*W, 1, C)
            # Per-token regression.
            attended = self.attn(readout)
            params = self.regress(attended)  # (B, H*W, out_channels)
            return params.transpose(1, 2).reshape(b, self.out_channels, h, w)
        if x.ndim == 4:
            attended = self.attn(x)
            return self.regress(attended)
        raise ValueError(
            f"expected input of shape (B, C, H, W) or (B, N, T, C); got {tuple(x.shape)}."
        )
