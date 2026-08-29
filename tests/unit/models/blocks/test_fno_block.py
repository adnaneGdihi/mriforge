"""Tests for ``SpectralConv2d`` and ``FNOBlock``.

Targets ``mriforge.models.blocks.fno_block``.

Categories:

- ``SpectralConv2d``: shape preserved (output H, W same as input)
- ``SpectralConv2d``: weights are ``nn.Parameter`` (trainable)
- ``SpectralConv2d``: gradient flows back to input
- ``FNOBlock``: shape preserved (residual connection means same dims)
- ``FNOBlock``: gradient flows through both spectral + residual paths
- ``modes`` larger than the input H/W are silently capped
"""

from __future__ import annotations

import torch

from mriforge.models.blocks.fno_block import FNOBlock, SpectralConv2d


# ---------------------------------------------------------------------------
# SpectralConv2d
# ---------------------------------------------------------------------------


def test_spectral_conv_shape_preserved() -> None:
    """``SpectralConv2d`` returns ``[B, out_channels, H, W]``."""
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=8, modes2=8)
    x = torch.randn(2, 4, 16, 16)
    out = layer(x)
    assert out.shape == (2, 4, 16, 16)


def test_spectral_conv_modes_capped_to_input_size() -> None:
    """Calling with H/W < requested modes silently caps."""
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=128, modes2=128)
    x = torch.randn(1, 4, 8, 8)
    # Should not raise; modes capped internally.
    out = layer(x)
    assert out.shape == (1, 4, 8, 8)


def test_spectral_conv_weights_are_parameters() -> None:
    """Both ``weights1`` and ``weights2`` are learnable parameters."""
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=4, modes2=4)
    assert isinstance(layer.weights1, torch.nn.Parameter)
    assert isinstance(layer.weights2, torch.nn.Parameter)
    assert layer.weights1.requires_grad
    assert layer.weights2.requires_grad


def test_spectral_conv_complex_weight_dtype() -> None:
    """Weights are complex-valued (`torch.cfloat`-like)."""
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=4, modes2=4)
    assert torch.is_complex(layer.weights1)
    assert torch.is_complex(layer.weights2)


def test_spectral_conv_gradient_flows_to_input() -> None:
    """Backward through the spectral conv reaches the input tensor."""
    layer = SpectralConv2d(in_channels=4, out_channels=4, modes1=4, modes2=4)
    x = torch.randn(1, 4, 8, 8, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# FNOBlock
# ---------------------------------------------------------------------------


def test_fno_block_shape_preserved() -> None:
    """``FNOBlock`` preserves the input spatial shape."""
    block = FNOBlock(width=4, modes1=4, modes2=4)
    x = torch.randn(1, 4, 16, 16)
    out = block(x)
    assert out.shape == (1, 4, 16, 16)


def test_fno_block_gradient_flows_to_input() -> None:
    """Backward reaches the input via both spectral + residual paths."""
    block = FNOBlock(width=4, modes1=4, modes2=4)
    x = torch.randn(1, 4, 16, 16, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fno_block_gradient_flows_to_residual_conv() -> None:
    """Gradient flows into the 1×1 residual conv weights."""
    block = FNOBlock(width=4, modes1=4, modes2=4)
    x = torch.randn(1, 4, 16, 16)
    block(x).sum().backward()
    assert block.residual.weight.grad is not None
    assert torch.isfinite(block.residual.weight.grad).all()


def test_fno_block_gradient_flows_to_spectral_weights() -> None:
    """Gradient flows into the spectral-conv weight parameters."""
    block = FNOBlock(width=4, modes1=4, modes2=4)
    x = torch.randn(1, 4, 16, 16)
    block(x).sum().backward()
    assert block.conv.weights1.grad is not None
