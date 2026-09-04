"""Sanity-shape + canary tests for ``kspace_swin_attention``.

Targets:
- ``RadialWindowAttention`` — radial-window multi-head self-attention.
- ``KSpaceRadialSwinBlock`` — full Swin-style block (attention + MLP).

Forward contracts
-----------------
``RadialWindowAttention(dim, window_size, num_heads)``:
  input  ``(B, L, dim)`` + ``H: int`` + ``W: int`` where ``L == H * W``
  output ``(B, L, dim)``

``KSpaceRadialSwinBlock(dim, num_heads, window_size)``:
  input  ``(B, L, dim)`` + ``H`` + ``W``
  output ``(B, L, dim)``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.kspace_swin_attention import (
    KSpaceRadialSwinBlock,
    RadialWindowAttention,
)


# ── RadialWindowAttention ────────────────────────────────────────────────


class TestRadialWindowAttentionCanary:
    def test_canary_forward_shape(self) -> None:
        """Basic (B, H*W, C) → same shape."""
        H, W, C = 8, 8, 16
        attn = RadialWindowAttention(dim=C, window_size=16, num_heads=4)
        attn.eval()
        x = torch.randn(1, H * W, C)
        out = attn(x, H=H, W=W)
        assert out.shape == (1, H * W, C)
        assert torch.isfinite(out).all()

    def test_canary_output_is_finite(self) -> None:
        H, W, C = 4, 4, 8
        attn = RadialWindowAttention(dim=C, window_size=8, num_heads=2)
        attn.eval()
        x = torch.randn(2, H * W, C)
        out = attn(x, H=H, W=W)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("B,H,W,C,ws,nh", [
    (1, 4, 4, 8, 8, 2),
    (2, 8, 8, 16, 16, 4),
    (1, 8, 8, 16, 32, 4),  # window_size > L → pads to 1 window
    (1, 16, 16, 16, 64, 4),
], ids=["B1H4W4C8", "B2H8W8C16", "B1H8W8C16ws32", "B1H16W16C16"])
def test_radial_window_attention_shape_matrix(
    B: int, H: int, W: int, C: int, ws: int, nh: int
) -> None:
    attn = RadialWindowAttention(dim=C, window_size=ws, num_heads=nh)
    attn.eval()
    x = torch.randn(B, H * W, C)
    out = attn(x, H=H, W=W)
    assert out.shape == (B, H * W, C)
    assert torch.isfinite(out).all()


def test_radial_window_attention_seq_len_mismatch_raises() -> None:
    """L != H*W raises ``AssertionError``."""
    attn = RadialWindowAttention(dim=8, window_size=8, num_heads=2)
    x = torch.randn(1, 15, 8)  # L=15, but H=4, W=4 → H*W=16
    with pytest.raises(AssertionError):
        attn(x, H=4, W=4)


# ── KSpaceRadialSwinBlock ─────────────────────────────────────────────────


class TestKSpaceRadialSwinBlockCanary:
    def test_canary_no_shift(self) -> None:
        """Unshifted block returns same shape."""
        H, W, C = 8, 8, 16
        block = KSpaceRadialSwinBlock(dim=C, num_heads=4, window_size=16)
        block.eval()
        x = torch.randn(1, H * W, C)
        out = block(x, H=H, W=W)
        assert out.shape == (1, H * W, C)
        assert torch.isfinite(out).all()

    def test_canary_shifted(self) -> None:
        """Shifted block (SR-WMSA) returns same shape."""
        H, W, C = 8, 8, 16
        block = KSpaceRadialSwinBlock(
            dim=C, num_heads=4, window_size=16, shift_size=8
        )
        block.eval()
        x = torch.randn(1, H * W, C)
        out = block(x, H=H, W=W)
        assert out.shape == (1, H * W, C)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("B,H,W,C,ws,nh,shift", [
    (1, 4, 4, 8, 8, 2, 0),
    (2, 8, 8, 16, 16, 4, 0),
    (1, 8, 8, 16, 16, 4, 8),   # shifted
    (1, 16, 8, 16, 32, 4, 0),
], ids=["B1H4W4C8", "B2H8W8C16", "B1H8W8shifted", "B1H16W8C16"])
def test_kspace_swin_block_shape_matrix(
    B: int, H: int, W: int, C: int, ws: int, nh: int, shift: int
) -> None:
    block = KSpaceRadialSwinBlock(
        dim=C, num_heads=nh, window_size=ws, shift_size=shift
    )
    block.eval()
    x = torch.randn(B, H * W, C)
    out = block(x, H=H, W=W)
    assert out.shape == (B, H * W, C)
    assert torch.isfinite(out).all()


def test_kspace_swin_block_wrong_channel_raises() -> None:
    """Tensor with wrong channel dim fails inside LayerNorm / projection."""
    block = KSpaceRadialSwinBlock(dim=16, num_heads=4, window_size=8)
    block.eval()
    x = torch.randn(1, 16, 32)  # C=32 but block expects C=16
    with pytest.raises((RuntimeError, Exception)):
        block(x, H=4, W=4)
