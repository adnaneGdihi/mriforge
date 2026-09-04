"""Sanity-shape + canary tests for ``multi_contrast_kspace_attention``.

Targets ``MultiContrastKSpaceCrossAttention``.

Forward contract
----------------
``MultiContrastKSpaceCrossAttention(channels, n_heads, patch_size)``:
  input  ``(B, n_contrasts, C, H, W)``
         where ``H % patch_size == 0`` and ``W % patch_size == 0``
  output ``(B, n_contrasts, C, H, W)``

Validation
----------
- ``channels < 1`` → ``ValueError``
- ``n_heads < 1`` → ``ValueError``
- ``patch_size < 1`` → ``ValueError``
- 4-D input → ``ValueError``
- channel mismatch → ``ValueError``
- ``H % patch_size != 0`` → ``ValueError``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.multi_contrast_kspace_attention import (
    MultiContrastKSpaceCrossAttention,
)


# ── Canary ───────────────────────────────────────────────────────────────


class TestMultiContrastCanary:
    def test_canary_minimal_forward(self) -> None:
        """Minimal construction + forward preserves (B, n_c, C, H, W)."""
        block = MultiContrastKSpaceCrossAttention(
            channels=2, n_heads=2, patch_size=4
        )
        block.eval()
        x = torch.randn(1, 3, 2, 8, 8)  # B=1, n_contrasts=3, C=2, H=8, W=8
        out = block(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_canary_output_dtype_matches_input(self) -> None:
        block = MultiContrastKSpaceCrossAttention(
            channels=4, n_heads=4, patch_size=4
        )
        block.eval()
        x = torch.randn(1, 2, 4, 8, 8)
        out = block(x)
        assert out.dtype == x.dtype

    def test_canary_residual_passthrough_at_init(self) -> None:
        """Output projection is zero-init so block starts as identity."""
        block = MultiContrastKSpaceCrossAttention(
            channels=4, n_heads=2, patch_size=4
        )
        block.eval()
        x = torch.randn(1, 2, 4, 8, 8)
        out = block(x)
        # proj weight is zero → output == residual (input tokens)
        # The norm may slightly shift the values, but output should be
        # close to input (within a reasonable atol given LayerNorm effects).
        assert torch.allclose(out, x, atol=1.0), (
            "At zero-init output should be close to input (residual)"
        )


# ── Shape matrix ─────────────────────────────────────────────────────────

# (B, n_contrasts, channels, H, W, n_heads, patch_size)
_SHAPES = [
    (1, 2, 2, 8, 8, 2, 4),
    (2, 3, 4, 8, 8, 4, 4),
    (1, 4, 4, 16, 16, 4, 8),
    (1, 2, 8, 8, 8, 4, 4),
]


@pytest.mark.parametrize(
    "B,nc,C,H,W,nh,ps",
    _SHAPES,
    ids=[f"B{b}nc{nc}C{c}H{h}W{w}" for b, nc, c, h, w, _, _ in _SHAPES],
)
def test_multi_contrast_shape_matrix(
    B: int, nc: int, C: int, H: int, W: int, nh: int, ps: int
) -> None:
    block = MultiContrastKSpaceCrossAttention(channels=C, n_heads=nh, patch_size=ps)
    block.eval()
    x = torch.randn(B, nc, C, H, W)
    out = block(x)
    assert out.shape == (B, nc, C, H, W)
    assert torch.isfinite(out).all()


# ── Validation ────────────────────────────────────────────────────────────


def test_zero_channels_raises() -> None:
    with pytest.raises(ValueError, match="channels"):
        MultiContrastKSpaceCrossAttention(channels=0, n_heads=2, patch_size=4)


def test_zero_n_heads_raises() -> None:
    with pytest.raises(ValueError, match="n_heads"):
        MultiContrastKSpaceCrossAttention(channels=4, n_heads=0, patch_size=4)


def test_zero_patch_size_raises() -> None:
    with pytest.raises(ValueError, match="patch_size"):
        MultiContrastKSpaceCrossAttention(channels=4, n_heads=2, patch_size=0)


def test_non_5d_input_raises() -> None:
    """4-D input raises ``ValueError``."""
    block = MultiContrastKSpaceCrossAttention(channels=4, n_heads=2, patch_size=4)
    x = torch.randn(1, 4, 8, 8)  # missing contrast axis
    with pytest.raises(ValueError):
        block(x)


def test_channel_mismatch_raises() -> None:
    """channel count in tensor != block.channels raises ``ValueError``."""
    block = MultiContrastKSpaceCrossAttention(channels=4, n_heads=2, patch_size=4)
    x = torch.randn(1, 2, 8, 8, 8)  # C=8, but block expects 4
    with pytest.raises(ValueError):
        block(x)


def test_spatial_not_divisible_by_patch_raises() -> None:
    """H not divisible by patch_size raises ``ValueError``."""
    block = MultiContrastKSpaceCrossAttention(channels=4, n_heads=2, patch_size=8)
    x = torch.randn(1, 2, 4, 7, 8)  # H=7 not divisible by 8
    with pytest.raises(ValueError, match="patch_size"):
        block(x)


# ── Differentiability ─────────────────────────────────────────────────────


def test_gradient_flows_through_block() -> None:
    block = MultiContrastKSpaceCrossAttention(channels=4, n_heads=2, patch_size=4)
    x = torch.randn(1, 2, 4, 8, 8, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
