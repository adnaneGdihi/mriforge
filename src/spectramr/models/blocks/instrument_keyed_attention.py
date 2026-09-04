"""Frame fusion keyed on an exogenous instrument rather than on the anatomy.

Multi-frame fusion has to decide, per location, how much to trust each frame.
Standard cross-attention derives queries and keys from the anatomy features
themselves, which conflates two different questions: *is this frame well
registered here* and *does this anatomy resemble that anatomy*. On data where
frames differ by an unknown rigid offset, the first is the question that
matters and the second is a confound.

A virtual fiducial separates them. The marker is **exogenous**: identical in
every frame up to the same geometric transform the anatomy underwent, and
statistically independent of the anatomy. Any difference between two marker
frames is therefore purely geometric. Deriving queries and keys from the marker
while taking values from the anatomy makes the attention map a function of
registration quality alone, which is what identifies the fusion weights.

The three ``keys`` settings share one input contract, so an ablation over them
moves exactly one variable:

``content``
    Queries, keys and values all from the anatomy. The standard baseline.
``marker``
    Queries and keys from the instrument, values from the anatomy. Identical
    parameter count to ``content``.
``none``
    No attention. Frames are averaged uniformly. The mechanism-absent control;
    it necessarily carries fewer parameters than the other two, and an arm
    comparing against it must say so.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from spectramr.models.blocks.block_registry import register_block

AttentionKeys = Literal["content", "marker", "none"]

__all__ = ["AttentionKeys", "InstrumentKeyedCrossAttention"]


class _FrameAttention(nn.Module):
    """Per-pixel multi-head attention across the frame axis."""

    def __init__(self, channels: int, n_frames: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        # Frames are an unordered set to the attention, so without this the
        # module cannot tell frame 0 from frame 7 and the per-frame
        # shift-conditioning carries no addressable identity.
        self.frame_embed = nn.Parameter(torch.zeros(n_frames, channels))
        self.dropout = nn.Dropout(dropout)
        # Identity-at-init: ``out_proj`` starts at zero, so with the residual
        # below the block returns the frame mean on step 0 and the attention has
        # to EARN its contribution. Measured across eight exp-11 arms, a block
        # whose output norm starts 7-190x its input norm never recovers.
        nn.init.zeros_(self.out_proj.weight)

    def split_heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.view(-1, n, self.heads, self.head_dim).transpose(1, 2)

    def weights(self, src: torch.Tensor, n: int) -> torch.Tensor:
        """Softmax attention ``[B*H*W, heads, N, N]`` from key-source tokens."""
        q = self.split_heads(self.q_proj(src), n)
        k = self.split_heads(self.k_proj(src), n)
        return torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)


@register_block("instrument_keyed_attention")
class InstrumentKeyedCrossAttention(nn.Module):
    """Attention-weighted fusion of ``n_frames`` feature maps.

    Args:
        channels: Feature channels per frame.
        n_frames: Number of frames fused.
        keys: Where queries and keys come from. Unknown values raise
            (CLAUDE.md #9) rather than degrading to ``content``.
        heads: Attention heads. Must divide ``channels``.
        dropout: Dropout on the attention weights.

    Shape:
        - frames: ``[B, N, C, H, W]``
        - instrument: ``[B, N, C, H, W]``, required iff ``keys='marker'``
        - output: ``[B, C, H, W]``
    """

    def __init__(
        self,
        channels: int,
        n_frames: int,
        keys: AttentionKeys = "content",
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if keys not in ("content", "marker", "none"):
            raise ValueError(
                f"keys={keys!r} is not one of 'content' / 'marker' / 'none'. An "
                "unknown routing must raise, not silently fall back to "
                "content-keyed attention, or an arm would report an ablation it "
                "never ran."
            )
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.n_frames = n_frames
        self.keys: AttentionKeys = keys
        # Absent, not disabled: a "switched-off" attention that still ran its
        # projections would be a different model, not an ablation of one.
        self.attn = None if keys == "none" else _FrameAttention(channels, n_frames, heads, dropout)

    def _key_source(self, frames: torch.Tensor, instrument: torch.Tensor | None) -> torch.Tensor:
        if self.keys == "content":
            return frames
        if instrument is None:
            raise ValueError(
                "keys='marker' needs the instrument features, and none were "
                "given. The marker is what makes the attention map a function of "
                "geometry alone; without it this would silently become "
                "content-keyed attention under a marker-keyed label."
            )
        if instrument.shape != frames.shape:
            raise ValueError(
                f"instrument {tuple(instrument.shape)} must match frames "
                f"{tuple(frames.shape)}: it rides the same geometry."
            )
        return instrument

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        """``[B, N, C, H, W]`` -> ``[B*H*W, N, C]``, one token set per pixel."""
        b, n, c, h, w = x.shape
        return x.permute(0, 3, 4, 1, 2).reshape(b * h * w, n, c)

    def _check(self, frames: torch.Tensor) -> None:
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B, N, C, H, W], got {tuple(frames.shape)}")
        if frames.shape[1] != self.n_frames:
            raise ValueError(
                f"built for n_frames={self.n_frames}, got {frames.shape[1]}. The "
                "frame embedding is per-frame, so the count is a contract."
            )

    def attention_weights(
        self, frames: torch.Tensor, instrument: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-location frame-to-frame attention ``[B, heads, N, N, H, W]``.

        Exposed so a mechanism-fires probe can assert that marker-keyed and
        content-keyed routing produce genuinely DIFFERENT maps on the same
        input. A green forward pass only proves the block did not crash.
        """
        if self.attn is None:
            raise ValueError("keys='none' computes no attention weights")
        self._check(frames)
        b, n, _c, h, w = frames.shape
        src = self._to_tokens(self._key_source(frames, instrument))
        attn = self.attn.weights(src + self.attn.frame_embed, n)
        return attn.reshape(b, h, w, -1, n, n).permute(0, 3, 4, 5, 1, 2)

    def forward(self, frames: torch.Tensor, instrument: torch.Tensor | None = None) -> torch.Tensor:
        """Fuse ``frames`` into a single ``[B, C, H, W]`` feature map."""
        self._check(frames)
        if self.attn is None:
            return frames.mean(dim=1)

        b, n, c, h, w = frames.shape
        src = self._to_tokens(self._key_source(frames, instrument))
        val = self._to_tokens(frames)
        attn = self.attn.dropout(self.attn.weights(src + self.attn.frame_embed, n))
        out = torch.matmul(attn, self.attn.split_heads(self.attn.v_proj(val), n))
        out = out.transpose(1, 2).reshape(b * h * w, n, c)
        # Residual around the attention, so a zero-initialised out_proj makes
        # the whole block an identity-to-the-frame-mean at step 0.
        fused = (val + self.attn.out_proj(out)).mean(dim=1)
        return fused.view(b, h, w, c).permute(0, 3, 1, 2)
