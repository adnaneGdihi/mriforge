"""Tests for convolutional building blocks.

Targets ``mriforge.models.blocks.convolutional``. Common 2D / 3D
convolutional blocks used across U-Net / generator architectures.

Categories:

- ``ConvBlock2D``: optional norm / activation
- ``DoubleConv``: U-Net style double-conv with GroupNorm + LeakyReLU
- ``ConvBlock3D`` / ``DoubleConv3D``: 3D variants
- ``ResConvBlock``: residual connection preserves identity for zero output

Cherry-picked classes — full coverage of MAML-conditioned variants
is deferred to integration tier.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.models.blocks.convolutional import (
    ConvBlock2D,
    ConvBlock3D,
    DoubleConv,
    DoubleConv3D,
    ResConvBlock,
)


# ---------------------------------------------------------------------------
# ConvBlock2D
# ---------------------------------------------------------------------------


def test_conv_block_2d_default_output_shape() -> None:
    """``ConvBlock2D(3 → 8, kernel=3, padding=1)`` preserves spatial dims."""
    block = ConvBlock2D(3, 8)
    x = torch.randn(2, 3, 16, 16)
    out = block(x)
    assert out.shape == (2, 8, 16, 16)


def test_conv_block_2d_optional_norm_layer() -> None:
    """When ``norm_layer`` is ``BatchNorm2d``, it's applied after conv."""
    block = ConvBlock2D(3, 8, norm_layer=nn.BatchNorm2d)
    assert isinstance(block.norm, nn.BatchNorm2d)


def test_conv_block_2d_optional_activation_none() -> None:
    """``activation=None`` disables activation."""
    block = ConvBlock2D(3, 8, activation=None)
    assert block.activation is None


def test_conv_block_2d_stride_downsamples() -> None:
    """Stride > 1 reduces spatial dims."""
    block = ConvBlock2D(1, 4, kernel_size=3, stride=2, padding=1)
    x = torch.randn(1, 1, 16, 16)
    out = block(x)
    assert out.shape == (1, 4, 8, 8)


# ---------------------------------------------------------------------------
# DoubleConv (U-Net)
# ---------------------------------------------------------------------------


def test_double_conv_output_shape() -> None:
    """``DoubleConv(1 → 8)`` outputs ``[B, 8, H, W]``."""
    block = DoubleConv(1, 8)
    x = torch.randn(1, 1, 16, 16)
    out = block(x)
    assert out.shape == (1, 8, 16, 16)


def test_double_conv_uses_mid_channels() -> None:
    """Specifying ``mid_channels`` adjusts the inner conv's output."""
    block = DoubleConv(1, 8, mid_channels=4)
    # The first inner conv should produce 4 channels.
    inner_first_conv = block.double_conv[0]
    assert inner_first_conv.out_channels == 4


def test_double_conv_default_mid_channels() -> None:
    """Without ``mid_channels``, mid = out."""
    block = DoubleConv(1, 8)
    inner_first_conv = block.double_conv[0]
    assert inner_first_conv.out_channels == 8


def test_double_conv_gradient_flows() -> None:
    """Backward through DoubleConv reaches the input."""
    block = DoubleConv(1, 8)
    x = torch.randn(1, 1, 16, 16, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# ConvBlock3D
# ---------------------------------------------------------------------------


def test_conv_block_3d_output_shape() -> None:
    """``ConvBlock3D`` preserves spatial dims with padding=1."""
    block = ConvBlock3D(1, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 1, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 4, 4, 8, 8)


def test_double_conv_3d_output_shape() -> None:
    """``DoubleConv3D(1 → 4)`` outputs ``[B, 4, D, H, W]``."""
    block = DoubleConv3D(1, 4)
    x = torch.randn(1, 1, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 4, 4, 8, 8)


# ---------------------------------------------------------------------------
# ResConvBlock — residual connection
# ---------------------------------------------------------------------------


def test_res_conv_block_output_shape() -> None:
    """ResConvBlock preserves channel count and spatial shape.

    Note: the current constructor takes a single ``channels`` arg —
    in-channels == out-channels by construction.  See
    :func:`test_res_conv_block_channel_count_matches_input` for the
    contract this enforces.
    """
    block = ResConvBlock(channels=8)
    x = torch.randn(1, 8, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_res_conv_block_with_channel_change() -> None:
    """ResConvBlock takes a single ``channels`` argument; channel changes
    are not supported by this block.  Use :class:`ConvBlock` chained
    with a 1x1 projection for explicit channel changes instead.
    """
    import pytest
    with pytest.raises(TypeError):
        ResConvBlock(in_channels=4, out_channels=8)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_c, out_c, h, w",
    [(1, 8, 16, 16), (3, 16, 32, 32), (8, 4, 64, 32), (1, 64, 8, 8)],
    ids=["1to8_16", "3to16_32", "8to4_rect", "1to64_8"],
)
def test_double_conv_shape_matrix(in_c: int, out_c: int, h: int, w: int) -> None:
    """DoubleConv handles a spread of channel/spatial configs."""
    block = DoubleConv(in_c, out_c)
    x = torch.randn(1, in_c, h, w)
    out = block(x)
    assert out.shape == (1, out_c, h, w)
