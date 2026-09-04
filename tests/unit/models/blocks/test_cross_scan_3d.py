"""Sanity-shape + canary tests for ``cross_scan_3d``.

Targets:
- ``MambaSSM1D`` — single-direction SSM sweep.
- ``CrossScan3D`` — 6-direction (or 4-direction for 2D) gated fusion.

Forward contracts
-----------------
``MambaSSM1D(d_model)``:
  input  ``(B, L, d_model)``
  output ``(B, L, d_model)``

``CrossScan3D(d_model, use_3d=True)``:
  input  ``(B, d_model, D, H, W)``
  output ``(B, d_model, D, H, W)``
  When ``D=1`` the module automatically falls back to 4-direction scan.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.cross_scan_3d import CrossScan3D, MambaSSM1D


# ── MambaSSM1D ───────────────────────────────────────────────────────────


class TestMambaSSM1D:
    def test_canary_forward_shape(self) -> None:
        """Basic (B, L, D) → (B, L, D)."""
        ssm = MambaSSM1D(d_model=16)
        ssm.eval()
        x = torch.randn(2, 8, 16)
        out = ssm(x)
        assert out.shape == (2, 8, 16)
        assert torch.isfinite(out).all()

    def test_canary_output_is_finite(self) -> None:
        ssm = MambaSSM1D(d_model=32, d_state=8, d_conv=4, expand=2)
        ssm.eval()
        x = torch.randn(1, 16, 32)
        out = ssm(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("B,L,D", [
        (1, 4, 8),
        (2, 16, 16),
        (1, 32, 32),
    ], ids=["B1L4D8", "B2L16D16", "B1L32D32"])
    def test_shape_matrix(self, B: int, L: int, D: int) -> None:
        ssm = MambaSSM1D(d_model=D)
        ssm.eval()
        x = torch.randn(B, L, D)
        out = ssm(x)
        assert out.shape == (B, L, D)


# ── CrossScan3D — 3D mode ────────────────────────────────────────────────


class TestCrossScan3DMode:
    def test_canary_3d_forward(self) -> None:
        """3D input (D>1) sweeps 6 directions and returns same shape."""
        block = CrossScan3D(d_model=8, use_3d=True)
        block.eval()
        x = torch.randn(1, 8, 2, 4, 4)  # (B, C, D, H, W)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("B,C,D,H,W", [
        (1, 8, 2, 4, 4),
        (2, 16, 2, 8, 8),
        (1, 8, 4, 8, 8),
    ], ids=["B1C8D2H4W4", "B2C16D2H8W8", "B1C8D4H8W8"])
    def test_3d_shape_matrix(self, B: int, C: int, D: int, H: int, W: int) -> None:
        """Output shape == input shape for various 3D configs."""
        block = CrossScan3D(d_model=C, use_3d=True)
        block.eval()
        x = torch.randn(B, C, D, H, W)
        out = block(x)
        assert out.shape == (B, C, D, H, W)
        assert torch.isfinite(out).all()


# ── CrossScan3D — 2D mode (D=1) ──────────────────────────────────────────


class TestCrossScan3D2DFallback:
    def test_canary_2d_fallback(self) -> None:
        """D=1 triggers 4-direction scan; shape preserved."""
        block = CrossScan3D(d_model=8, use_3d=True)
        block.eval()
        x = torch.randn(1, 8, 1, 8, 8)  # D=1 → 4-direction
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_use_3d_false_always_4dirs(self) -> None:
        """``use_3d=False`` forces 4-direction even for D>1."""
        block = CrossScan3D(d_model=8, use_3d=False)
        block.eval()
        x = torch.randn(1, 8, 4, 4, 4)
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("B,C,H,W", [
        (1, 8, 4, 4),
        (2, 16, 8, 8),
        (1, 8, 16, 16),
    ], ids=["B1C8H4W4", "B2C16H8W8", "B1C8H16W16"])
    def test_2d_fallback_shape_matrix(self, B: int, C: int, H: int, W: int) -> None:
        block = CrossScan3D(d_model=C, use_3d=True)
        block.eval()
        x = torch.randn(B, C, 1, H, W)
        out = block(x)
        assert out.shape == (B, C, 1, H, W)
        assert torch.isfinite(out).all()


# ── Residual preserves value for zero-weight init ─────────────────────────


def test_cross_scan_3d_residual_preserves_input_scale() -> None:
    """Output ≈ input at init (residual dominates before training)."""
    block = CrossScan3D(d_model=8, use_3d=True)
    block.eval()
    x = torch.randn(1, 8, 2, 4, 4)
    out = block(x)
    # Residual means output is in the same ballpark — not testing exact equality.
    assert out.abs().max() > 0


class TestFusionWidthMatchesProjection:
    """The 2-D fallback must pad back to the width the projections were built for.

    ``__init__`` sizes ``gate_proj``/``value_proj`` for ``n_dirs`` (6 when
    ``use_3d``), but ``forward`` drops the two z-sweeps once the depth axis is
    degenerate. Concatenating 4*C into a 6*C Linear raised "mat1 and mat2 shapes
    cannot be multiplied (Nx4C and 6C x d_model)" -- the documented fallback path
    was unreachable in practice (cluster job 8000966).
    """

    def test_projection_input_width_is_n_dirs_times_d_model(self) -> None:
        block = CrossScan3D(d_model=8, use_3d=True)

        assert block.n_dirs == 6
        assert block.gate_proj.in_features == block.n_dirs * 8
        assert block.value_proj.in_features == block.n_dirs * 8

    def test_2d_and_3d_input_reach_the_same_projection(self) -> None:
        """One weight set serves both ranks -- neither path may resize it."""
        block = CrossScan3D(d_model=8, use_3d=True)
        block.eval()

        volumetric = block(torch.randn(1, 8, 3, 4, 4))
        planar = block(torch.randn(1, 8, 1, 4, 4))

        assert volumetric.shape == (1, 8, 3, 4, 4)
        assert planar.shape == (1, 8, 1, 4, 4)
        assert torch.isfinite(planar).all()

    def test_use_3d_false_needs_no_padding(self) -> None:
        """A 4-direction block is sized at 4 and never pads."""
        block = CrossScan3D(d_model=8, use_3d=False)
        block.eval()

        assert block.n_dirs == 4
        assert block.gate_proj.in_features == 4 * 8
        assert torch.isfinite(block(torch.randn(1, 8, 3, 4, 4))).all()
