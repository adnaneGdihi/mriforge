"""4-D bidirectional Mamba blocks for spatiotemporal SFCs.

Wires the framework's existing :class:`MambaBlock` (which auto-detects
the real ``mamba_ssm`` kernel and falls back to a gated-conv/GRU
approximation when CUDA kernels are unavailable) along a 4-D SFC
permutation.

Provides:

* :class:`BiMamba4DBlock` — fMRI §1: gathers a ``(C, T, H, W)``
  feature volume along a per-sample 4-D SFC permutation, runs a
  bidirectional Mamba scan, and scatters back. Falls back to a
  Conv1d-only depthwise scan when ``mamba_ssm`` cannot be imported
  (controlled via ``use_real_mamba=False``).
* :class:`MRFMambaBlock4D` — MRF §1: identical wiring with an extra
  per-timepoint conformal-Riemann-map argument that pre-applies the
  rotation correction before the scan.

Both blocks accept the indices produced by
:class:`BeltramiSFCBlock` (spatial-only) plus a separate temporal
order; if either is absent the block degenerates to a raster scan
plus convolution.

References:
    [13] Liu et al., "Vision Mamba: Efficient visual representation
         learning with bidirectional SSM", *ICML*, 2024.
    [Gu2023] Gu & Dao, "Mamba: Linear-time sequence modeling with
        selective state spaces", *arXiv:2312.00752*, 2023.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.models.blocks.mamba_block import MambaBlock

logger = logging.getLogger(__name__)


def _flatten_along_sfc(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a ``[B, C, T, H, W]`` tensor into a ``[B, C, T·H·W]`` sequence."""
    B, C, T, H, W = x.shape
    flat = x.reshape(B, C, T * H * W)
    idx = indices.unsqueeze(1).expand(B, C, T * H * W)
    return flat.gather(2, idx)


def _scatter_along_sfc(
    seq: torch.Tensor, indices: torch.Tensor, shape: tuple[int, int, int]
) -> torch.Tensor:
    B, C, _ = seq.shape
    T, H, W = shape
    out = seq.new_zeros(B, C, T * H * W)
    idx = indices.unsqueeze(1).expand(B, C, T * H * W)
    out.scatter_(2, idx, seq)
    return out.view(B, C, T, H, W)


class BiMamba4DBlock(nn.Module):
    """Bidirectional Mamba scan over a 4-D SFC traversal.

    Args:
        channels: Feature channels.
        kernel_size: Used only by the ``use_real_mamba=False`` Conv1d
            fallback; ignored when the real Mamba kernel is active.
        d_state: Mamba SSM state dimension (default 16).
        use_real_mamba: When True (default), uses
            :class:`MambaBlock` which auto-detects ``mamba_ssm``;
            False falls back to depthwise Conv1d only (useful for
            unit tests where the SSM kernel adds noise).
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        d_state: int = 16,
        use_real_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_real_mamba = use_real_mamba
        if use_real_mamba:
            self.fwd = MambaBlock(d_model=channels, d_state=d_state, d_conv=kernel_size)
            self.bwd = MambaBlock(d_model=channels, d_state=d_state, d_conv=kernel_size)
        else:
            self.fwd = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
            )
            self.bwd = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
            )
        self.proj = nn.Conv1d(channels * 2, channels, 1)
        self.act = nn.GELU()

    def _scan(self, seq_btl: torch.Tensor, mamba: nn.Module) -> torch.Tensor:
        """Run a single (fwd or bwd) scan. ``seq_btl`` is ``[B, C, L]``.

        MambaBlock expects ``[B, L, D]`` while Conv1d wants ``[B, C, L]``.
        """
        if self.use_real_mamba:
            seq_bld = seq_btl.transpose(1, 2)  # [B, L, C]
            out_bld = mamba(seq_bld)
            return out_bld.transpose(1, 2)  # [B, C, L]
        return mamba(seq_btl)

    def forward(self, x: torch.Tensor, *, sfc_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Forward.

        Args:
            x: ``[B, C, T, H, W]`` feature tensor.
            sfc_indices: ``[B, T·H·W]`` permutation. Falls back to
                raster scan when absent.
        """
        if x.dim() != 5:
            raise ValueError(f"BiMamba4DBlock expects 5-D input, got {x.shape}")
        B, C, T, H, W = x.shape
        if sfc_indices is None:
            seq = x.reshape(B, C, T * H * W)
        else:
            seq = _flatten_along_sfc(x, sfc_indices)
        f = self._scan(seq, self.fwd)
        b = self._scan(seq.flip(-1), self.bwd).flip(-1)
        merged = self.act(self.proj(torch.cat([f, b], dim=1)))
        if sfc_indices is None:
            out = merged.view(B, C, T, H, W)
        else:
            out = _scatter_along_sfc(merged, sfc_indices, (T, H, W))
        return x + out


class MRFMambaBlock4D(BiMamba4DBlock):
    """4-D bidirectional Mamba with per-timepoint conformal pre-rotation.

    The ``rotation_angles`` tensor (one angle per timepoint) is applied
    as a phase rotation of the per-timepoint spatial features before
    the scan — corresponds to the spiral-arm rotation correction in
    MRF §1.

    Inherits the constructor signature of :class:`BiMamba4DBlock`
    including ``use_real_mamba`` / ``d_state``.
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        *,
        sfc_indices: torch.Tensor | None = None,
        rotation_angles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rotation_angles is not None and x.dim() == 5:
            B, C, T, H, W = x.shape
            if rotation_angles.shape[-1] != T:
                raise ValueError(f"rotation_angles length {rotation_angles.shape[-1]} != T={T}")
            cos = rotation_angles.cos().view(1, 1, T, 1, 1)
            sin = rotation_angles.sin().view(1, 1, T, 1, 1)
            # Simple 2-channel rotation of the first two channels as a
            # diagnostic / placeholder — full radial->disk mapping would
            # plug a precomputed grid in here.
            if C >= 2:
                ch0, ch1 = x[:, 0:1], x[:, 1:2]
                rot0 = ch0 * cos - ch1 * sin
                rot1 = ch0 * sin + ch1 * cos
                x = torch.cat([rot0, rot1, x[:, 2:]], dim=1)
        return super().forward(x, sfc_indices=sfc_indices)


__all__ = ["BiMamba4DBlock", "MRFMambaBlock4D"]
