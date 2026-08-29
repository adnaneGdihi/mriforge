"""Tests for ToeplitzSpectralAttention."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.toeplitz_attention import ToeplitzSpectralAttention


def test_forward_shape_preserved() -> None:
    block = ToeplitzSpectralAttention(embed_dim=16, num_heads=4, token_len=32)
    x = torch.randn(2, 32, 16)
    out = block(x)
    assert out.shape == x.shape


def test_complexity_is_n_log_n() -> None:
    """Reported FLOPs grow as N log N, not N²."""
    f_64 = ToeplitzSpectralAttention.expected_flops(64, 16)
    f_128 = ToeplitzSpectralAttention.expected_flops(128, 16)
    # Doubling N should give a less-than-4x increase (vs full attention).
    assert f_128 < 4 * f_64


def test_init_validation() -> None:
    with pytest.raises(ValueError):
        ToeplitzSpectralAttention(embed_dim=15, num_heads=4, token_len=32)
    with pytest.raises(ValueError):
        ToeplitzSpectralAttention(embed_dim=16, num_heads=4, token_len=1)


def test_forward_invalid_shape() -> None:
    block = ToeplitzSpectralAttention(embed_dim=16, num_heads=4, token_len=32)
    with pytest.raises(ValueError):
        block(torch.randn(2, 16))  # missing token axis
    with pytest.raises(ValueError):
        block(torch.randn(2, 33, 16))  # wrong token_len
    with pytest.raises(ValueError):
        block(torch.randn(2, 32, 8))  # wrong embed_dim


def test_gradient_flow() -> None:
    block = ToeplitzSpectralAttention(embed_dim=8, num_heads=2, token_len=16)
    x = torch.randn(1, 16, 8, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.allclose(block.tau.grad, torch.zeros_like(block.tau.grad))


def test_initial_kernel_is_normalised() -> None:
    block = ToeplitzSpectralAttention(embed_dim=4, num_heads=1, token_len=16)
    assert torch.allclose(
        block.tau[0].sum(), torch.tensor(1.0), atol=1e-4
    )
