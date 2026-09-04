"""Tests for complex-valued residual / NAFNet blocks.

Targets ``spectramr.models.blocks.complex_blocks``. ``ComplexResBlock`` and
``ComplexNAFBlock`` operate on real-stacked complex tensors
``[B, 2C, H, W]`` (R/I interleaved).

Categories:

- Construction validation: even-channel constraint
- Shape preservation
- Gradient flow
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.complex_blocks import ComplexNAFBlock, ComplexResBlock


# ---------------------------------------------------------------------------
# ComplexResBlock — construction
# ---------------------------------------------------------------------------


def test_complex_resblock_requires_even_in_channels() -> None:
    """Odd ``in_channels`` → ``ValueError``."""
    with pytest.raises(ValueError, match="even channels"):
        ComplexResBlock(in_channels=3, out_channels=8, emb_dim=16)


def test_complex_resblock_requires_even_out_channels() -> None:
    """Odd ``out_channels`` → ``ValueError``."""
    with pytest.raises(ValueError, match="even channels"):
        ComplexResBlock(in_channels=8, out_channels=5, emb_dim=16)


def test_complex_resblock_constructs_with_even_channels() -> None:
    """Even channels → construction succeeds."""
    block = ComplexResBlock(in_channels=8, out_channels=8, emb_dim=16)
    assert block is not None


# ---------------------------------------------------------------------------
# ComplexResBlock — forward
# ---------------------------------------------------------------------------


def test_complex_resblock_preserves_shape() -> None:
    """Input/output have the same spatial dims."""
    block = ComplexResBlock(in_channels=8, out_channels=8, emb_dim=16)
    x = torch.randn(1, 8, 16, 16)
    emb = torch.randn(1, 16)
    out = block(x, emb)
    assert out.shape == x.shape


def test_complex_resblock_changes_channels() -> None:
    """``in != out`` channels → 1×1 shortcut adapts dimensions."""
    block = ComplexResBlock(in_channels=8, out_channels=16, emb_dim=16)
    x = torch.randn(1, 8, 16, 16)
    emb = torch.randn(1, 16)
    out = block(x, emb)
    assert out.shape == (1, 16, 16, 16)


def test_complex_resblock_gradient_flows() -> None:
    """Backward through the block reaches the input."""
    block = ComplexResBlock(in_channels=8, out_channels=8, emb_dim=16)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    emb = torch.randn(1, 16)
    block(x, emb).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# ComplexNAFBlock — construction
# ---------------------------------------------------------------------------


def test_complex_nafblock_requires_even_channels() -> None:
    """Odd ``c`` → ``ValueError``."""
    with pytest.raises(ValueError, match="even channels"):
        ComplexNAFBlock(c=5)


def test_complex_nafblock_constructs_with_even_channels() -> None:
    """Even channel count succeeds."""
    block = ComplexNAFBlock(c=8)
    assert block is not None


def test_complex_nafblock_with_time_embedding() -> None:
    """Time-embedding projection is created when ``time_embedding_dim`` is set."""
    block = ComplexNAFBlock(c=8, time_embedding_dim=32)
    assert block.time_proj is not None


def test_complex_nafblock_no_time_embedding_by_default() -> None:
    """Without ``time_embedding_dim``, ``time_proj`` is ``None``."""
    block = ComplexNAFBlock(c=8)
    assert block.time_proj is None


# ---------------------------------------------------------------------------
# ComplexNAFBlock — forward
# ---------------------------------------------------------------------------


def test_complex_nafblock_preserves_shape() -> None:
    """Input/output share the same shape."""
    block = ComplexNAFBlock(c=8)
    x = torch.randn(1, 8, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_complex_nafblock_with_time_embedding_forward() -> None:
    """Time-embedded forward runs and preserves shape."""
    block = ComplexNAFBlock(c=8, time_embedding_dim=16)
    x = torch.randn(1, 8, 16, 16)
    emb = torch.randn(1, 16)
    out = block(x, emb)
    assert out.shape == x.shape


def test_complex_nafblock_initialises_layer_scale_to_zero() -> None:
    """``beta`` and ``gamma`` are initialised to zero (residual = identity initially)."""
    block = ComplexNAFBlock(c=8)
    assert torch.allclose(block.beta, torch.zeros_like(block.beta))
    assert torch.allclose(block.gamma, torch.zeros_like(block.gamma))


def test_complex_nafblock_initial_output_equals_input() -> None:
    """With β=γ=0 (default init), output = input (residual identity)."""
    block = ComplexNAFBlock(c=8)
    x = torch.randn(1, 8, 16, 16)
    out = block(x)
    assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels, h, w",
    [(8, 16, 16), (16, 32, 32), (4, 8, 16)],
    ids=["c8_16x16", "c16_32x32", "c4_8x16"],
)
def test_complex_resblock_shape_matrix(channels: int, h: int, w: int) -> None:
    """ComplexResBlock handles the parametrised shape matrix."""
    block = ComplexResBlock(in_channels=channels, out_channels=channels, emb_dim=8)
    x = torch.randn(1, channels, h, w)
    emb = torch.randn(1, 8)
    out = block(x, emb)
    assert out.shape == (1, channels, h, w)
