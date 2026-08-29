"""Miscellaneous sanity-shape + canary tests for smaller zero-cov blocks.

Groups:
- ``pseudo_3d_kspace``: Pseudo3DKSpaceBlock, RepetitionFusion
- ``sfc_windowed_attention``: SFCWindowedAttention, CrossDomainBlock

Each block gets at least one canary forward test (finite output, correct
shape) and one invalid-config raise test where the block declares it.
"""

from __future__ import annotations

import pytest
import torch


# ═══════════════════════════════════════════════════════════════════════
# pseudo_3d_kspace
# ═══════════════════════════════════════════════════════════════════════


from mriforge.models.blocks.pseudo_3d_kspace import (  # noqa: E402
    Pseudo3DKSpaceBlock,
    RepetitionFusion,
)


class TestPseudo3DKSpaceBlock:
    """Pseudo-3D convolution block that restricts spatial kernel to 1×1."""

    def test_canary_real_forward(self) -> None:
        """Real tensor: (B, C_in, D, H, W) → (B, C_out, D, H, W)."""
        block = Pseudo3DKSpaceBlock(in_channels=4, out_channels=8, kernel_depth=3)
        block.eval()
        x = torch.randn(1, 4, 4, 8, 8)
        out = block(x)
        assert out.shape == (1, 8, 4, 8, 8)
        assert torch.isfinite(out).all()

    def test_canary_same_channels(self) -> None:
        """in_channels == out_channels preserves spatial shape."""
        block = Pseudo3DKSpaceBlock(in_channels=8, out_channels=8)
        block.eval()
        x = torch.randn(2, 8, 2, 4, 4)
        out = block(x)
        assert out.shape == (2, 8, 2, 4, 4)
        assert torch.isfinite(out).all()

    def test_canary_no_norm_no_act(self) -> None:
        """Block with norm=False, activation=False still works."""
        block = Pseudo3DKSpaceBlock(
            in_channels=4, out_channels=4, activation=False, norm=False
        )
        block.eval()
        x = torch.randn(1, 4, 2, 4, 4)
        out = block(x)
        assert out.shape == (1, 4, 2, 4, 4)

    @pytest.mark.parametrize("B,Cin,Cout,D,H,W", [
        (1, 2, 4, 2, 4, 4),
        (2, 4, 8, 4, 8, 8),
        (1, 8, 8, 2, 16, 16),
    ], ids=["small", "mid", "large"])
    def test_shape_matrix(self, B: int, Cin: int, Cout: int, D: int, H: int, W: int) -> None:
        block = Pseudo3DKSpaceBlock(in_channels=Cin, out_channels=Cout)
        block.eval()
        x = torch.randn(B, Cin, D, H, W)
        out = block(x)
        assert out.shape == (B, Cout, D, H, W)
        assert torch.isfinite(out).all()

    def test_complex_forward(self) -> None:
        """Complex input processes real/imag separately and returns complex."""
        block = Pseudo3DKSpaceBlock(in_channels=4, out_channels=4, norm=False)
        block.eval()
        x = torch.randn(1, 4, 2, 4, 4).to(torch.complex64)
        out = block(x)
        assert out.dtype == torch.complex64
        assert out.shape == (1, 4, 2, 4, 4)
        assert torch.isfinite(out.real).all()
        assert torch.isfinite(out.imag).all()


class TestRepetitionFusion:
    """Attention-weighted repetition fusion module."""

    def test_canary_forward(self) -> None:
        """(B, C, H, W, Reps) → (B, C, H, W) after attention-weighted collapse."""
        block = RepetitionFusion(channels=4, num_reps=3, num_layers=1)
        block.eval()
        # TorchIO layout: (B, C, H, W, Reps)
        x = torch.randn(1, 4, 8, 8, 3)
        out = block(x)
        assert out.shape == (1, 4, 8, 8)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("B,C,H,W,R", [
        (1, 4, 4, 4, 2),
        (2, 8, 8, 8, 4),
    ], ids=["small", "mid"])
    def test_shape_matrix(self, B: int, C: int, H: int, W: int, R: int) -> None:
        block = RepetitionFusion(channels=C, num_reps=R, num_layers=1)
        block.eval()
        x = torch.randn(B, C, H, W, R)
        out = block(x)
        assert out.shape == (B, C, H, W)
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════
# sfc_windowed_attention
# ═══════════════════════════════════════════════════════════════════════


from mriforge.models.blocks.sfc_windowed_attention import (  # noqa: E402
    CrossDomainBlock,
    SFCWindowedAttention,
)


class TestSFCWindowedAttention:
    """Local-window attention along 1-D SFC ordering."""

    def test_canary_forward_shape(self) -> None:
        """(B, T, D) → (B, T, D)."""
        block = SFCWindowedAttention(d_model=16, n_heads=4, window=8, n_global=2)
        block.eval()
        x = torch.randn(1, 16, 16)
        out = block(x)
        assert out.shape == (1, 16, 16)
        assert torch.isfinite(out).all()

    def test_canary_no_global_tokens(self) -> None:
        """n_global=0 skips the global-token prepend path."""
        block = SFCWindowedAttention(d_model=8, n_heads=2, window=4, n_global=0)
        block.eval()
        x = torch.randn(2, 8, 8)
        out = block(x)
        assert out.shape == (2, 8, 8)
        assert torch.isfinite(out).all()

    def test_canary_seq_not_multiple_of_window(self) -> None:
        """T not divisible by window is right-padded internally."""
        block = SFCWindowedAttention(d_model=8, n_heads=2, window=4, n_global=1)
        block.eval()
        x = torch.randn(1, 7, 8)  # T=7, window=4 → pads to 8
        out = block(x)
        assert out.shape == (1, 7, 8)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("B,T,D,nh,win,ng", [
        (1, 8, 8, 2, 4, 0),
        (2, 16, 16, 4, 8, 4),
        (1, 32, 32, 8, 16, 8),
    ], ids=["B1T8D8", "B2T16D16", "B1T32D32"])
    def test_shape_matrix(self, B: int, T: int, D: int, nh: int, win: int, ng: int) -> None:
        block = SFCWindowedAttention(d_model=D, n_heads=nh, window=win, n_global=ng)
        block.eval()
        x = torch.randn(B, T, D)
        out = block(x)
        assert out.shape == (B, T, D)
        assert torch.isfinite(out).all()

    def test_wrong_d_model_raises(self) -> None:
        """Tensor with D != d_model raises ``ValueError``."""
        block = SFCWindowedAttention(d_model=8, n_heads=2, window=4, n_global=0)
        x = torch.randn(1, 4, 16)  # D=16, but block expects 8
        with pytest.raises(ValueError, match="Expected D="):
            block(x)

    def test_d_model_not_divisible_by_n_heads_raises(self) -> None:
        """d_model not divisible by n_heads raises ``ValueError`` at construction."""
        with pytest.raises(ValueError, match="divisible"):
            SFCWindowedAttention(d_model=7, n_heads=4)

    def test_window_zero_raises(self) -> None:
        """window=0 raises ``ValueError``."""
        with pytest.raises(ValueError):
            SFCWindowedAttention(d_model=8, n_heads=2, window=0)


class TestCrossDomainBlock:
    """Two parallel SFCWindowedAttention streams + optional cross-attention."""

    def test_canary_no_cross(self) -> None:
        """Without cross-attention, both streams are returned unchanged shape."""
        block = CrossDomainBlock(d_model=16, n_heads=4, win_attn=8, n_global=2, cross=False)
        block.eval()
        z_k = torch.randn(1, 8, 16)
        z_x = torch.randn(1, 8, 16)
        out_k, out_x = block(z_k, z_x)
        assert out_k.shape == (1, 8, 16)
        assert out_x.shape == (1, 8, 16)
        assert torch.isfinite(out_k).all()
        assert torch.isfinite(out_x).all()

    def test_canary_with_cross(self) -> None:
        """With cross-attention, shapes still preserved."""
        block = CrossDomainBlock(d_model=16, n_heads=4, win_attn=8, n_global=2, cross=True)
        block.eval()
        z_k = torch.randn(1, 8, 16)
        z_x = torch.randn(1, 8, 16)
        out_k, out_x = block(z_k, z_x)
        assert out_k.shape == (1, 8, 16)
        assert out_x.shape == (1, 8, 16)
        assert torch.isfinite(out_k).all()
        assert torch.isfinite(out_x).all()

    @pytest.mark.parametrize("cross", [False, True], ids=["no_cross", "with_cross"])
    def test_shape_matrix_cross_variants(self, cross: bool) -> None:
        block = CrossDomainBlock(
            d_model=16, n_heads=4, win_attn=8, n_global=0, cross=cross
        )
        block.eval()
        z_k = torch.randn(2, 16, 16)
        z_x = torch.randn(2, 16, 16)
        out_k, out_x = block(z_k, z_x)
        assert out_k.shape == (2, 16, 16)
        assert out_x.shape == (2, 16, 16)
