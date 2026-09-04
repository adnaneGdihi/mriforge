"""Sanity-shape + canary tests for ``graph_attention``.

Targets:
- ``GraphAttentionBlock`` — multi-head graph attention for k-space.
- ``LocalGraphAttentionBlock`` — KNN-constrained variant.

Forward contract
----------------
``GraphAttentionBlock(in_channels, out_channels, heads, concat_heads=True)``:
  input  ``x: (B, N, in_channels)`` *or* ``(B, in_channels, N)``
         ``trajectory: (B, 2, N)`` (optional)
         ``time_vec: (B, 1, N)`` (optional)
  output same shape as x after transposing back if needed;
         for concat_heads=True the last dim is ``out_channels``.

The block asserts ``out_channels % heads == 0`` when ``concat_heads=True``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.graph_attention import (
    GraphAttentionBlock,
    LocalGraphAttentionBlock,
)


# ── Canary ───────────────────────────────────────────────────────────────


class TestGraphAttentionBlockCanary:
    """Minimal-arg construction + forward."""

    def test_canary_bnc_input_forward_shape(self) -> None:
        """(B, N, C) input returns (B, N, out_channels)."""
        block = GraphAttentionBlock(in_channels=8, out_channels=8, heads=4)
        block.eval()
        x = torch.randn(2, 16, 8)  # (B, N, C)
        out = block(x)
        assert out.shape == (2, 16, 8)
        assert torch.isfinite(out).all()

    def test_canary_bcn_input_transposed_back(self) -> None:
        """(B, C, N) input is recognised and transposed back."""
        # C < N so the block detects channel-first format.
        block = GraphAttentionBlock(in_channels=4, out_channels=4, heads=2)
        block.eval()
        x = torch.randn(1, 4, 32)  # (B, C, N)
        out = block(x)
        # Should come back as (B, C, N)
        assert out.shape == (1, 4, 32)
        assert torch.isfinite(out).all()

    def test_canary_output_is_finite(self) -> None:
        block = GraphAttentionBlock(in_channels=16, out_channels=16, heads=4)
        block.eval()
        x = torch.randn(2, 10, 16)
        out = block(x)
        assert torch.isfinite(out).all()


# ── Shape matrix ─────────────────────────────────────────────────────────

# (B, N, in_channels, out_channels, heads)
_SHAPES = [
    (1, 8, 4, 4, 2),
    (2, 16, 8, 8, 4),
    (1, 32, 16, 16, 4),
    (2, 20, 8, 8, 2),
]


@pytest.mark.parametrize(
    "B,N,Cin,Cout,H",
    _SHAPES,
    ids=[f"B{b}N{n}Cin{ci}Cout{co}H{h}" for b, n, ci, co, h in _SHAPES],
)
def test_graph_attention_forward_shape(
    B: int, N: int, Cin: int, Cout: int, H: int
) -> None:
    """Forward returns ``(B, N, Cout)`` or ``(B, Cout, N)`` matching input fmt."""
    block = GraphAttentionBlock(in_channels=Cin, out_channels=Cout, heads=H)
    block.eval()
    x = torch.randn(B, N, Cin)  # (B, N, C)
    out = block(x)
    assert out.shape == (B, N, Cout)
    assert torch.isfinite(out).all()


# ── With optional inputs ─────────────────────────────────────────────────


def test_forward_with_trajectory() -> None:
    """Trajectory mask is applied without raising."""
    block = GraphAttentionBlock(
        in_channels=8, out_channels=8, heads=4, use_locality_mask=True
    )
    block.eval()
    B, N = 1, 12
    x = torch.randn(B, N, 8)
    traj = torch.randn(B, 2, N)
    out = block(x, trajectory=traj)
    assert out.shape == (B, N, 8)
    assert torch.isfinite(out).all()


def test_forward_with_time_vec() -> None:
    """Time-modulated attention completes without raising."""
    block = GraphAttentionBlock(in_channels=8, out_channels=8, heads=4)
    block.eval()
    B, N = 1, 8
    x = torch.randn(B, N, 8)
    time_vec = torch.rand(B, 1, N)
    out = block(x, time_vec=time_vec)
    assert out.shape == (B, N, 8)
    assert torch.isfinite(out).all()


def test_forward_with_all_optional_inputs() -> None:
    block = GraphAttentionBlock(
        in_channels=8, out_channels=8, heads=4, use_locality_mask=True
    )
    block.eval()
    B, N = 2, 10
    x = torch.randn(B, N, 8)
    traj = torch.randn(B, 2, N)
    time_vec = torch.rand(B, 1, N)
    out = block(x, trajectory=traj, time_vec=time_vec)
    assert out.shape == (B, N, 8)
    assert torch.isfinite(out).all()


# ── Invalid config ───────────────────────────────────────────────────────


def test_out_channels_not_divisible_by_heads_raises() -> None:
    """``out_channels % heads != 0`` with ``concat_heads=True`` raises."""
    with pytest.raises((AssertionError, ValueError)):
        GraphAttentionBlock(in_channels=8, out_channels=7, heads=4, concat_heads=True)


# ── Average-head variant ─────────────────────────────────────────────────


def test_concat_false_averages_heads() -> None:
    """``concat_heads=False`` works without divisibility constraint."""
    block = GraphAttentionBlock(
        in_channels=8, out_channels=8, heads=3, concat_heads=False
    )
    block.eval()
    x = torch.randn(1, 6, 8)
    out = block(x)
    assert out.shape == (1, 6, 8)
    assert torch.isfinite(out).all()


# ── LocalGraphAttentionBlock ─────────────────────────────────────────────


class TestLocalGraphAttentionBlock:
    def test_canary_small_graph_falls_back_to_global(self) -> None:
        """Small N < 256 falls back to global attention."""
        block = LocalGraphAttentionBlock(
            in_channels=8, out_channels=8, k_neighbors=8
        )
        block.eval()
        x = torch.randn(1, 16, 8)
        out = block(x)
        assert out.shape == (1, 16, 8)
        assert torch.isfinite(out).all()

    def test_canary_with_trajectory(self) -> None:
        block = LocalGraphAttentionBlock(
            in_channels=8, out_channels=8, k_neighbors=4
        )
        block.eval()
        B, N = 1, 20
        x = torch.randn(B, N, 8)
        traj = torch.randn(B, 2, N)
        out = block(x, trajectory=traj)
        assert out.shape == (B, N, 8)
        assert torch.isfinite(out).all()
