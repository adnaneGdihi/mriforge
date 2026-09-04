"""Tests for MatrixProductStateAttention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.mps_attention import MatrixProductStateAttention


def test_forward_shape_preserved() -> None:
    block = MatrixProductStateAttention(embed_dim=8, n_log_tokens=4, bond_dim=2)
    x = torch.randn(2, 16, 8)
    out = block(x)
    assert out.shape == x.shape


def test_token_len_validation() -> None:
    block = MatrixProductStateAttention(embed_dim=8, n_log_tokens=4)
    with pytest.raises(ValueError):
        block(torch.randn(1, 15, 8))  # 15 is not 2^4
    with pytest.raises(ValueError):
        block(torch.randn(1, 16, 5))


def test_bond_dim_validation() -> None:
    with pytest.raises(ValueError):
        MatrixProductStateAttention(embed_dim=8, n_log_tokens=4, bond_dim=0)
    with pytest.raises(ValueError):
        MatrixProductStateAttention(embed_dim=8, n_log_tokens=0)


def test_parameter_count_scales_with_bond_dim() -> None:
    """Parameter count is O(L χ² d²) — linear in L, quadratic in χ."""
    block_small = MatrixProductStateAttention(embed_dim=4, n_log_tokens=4, bond_dim=2)
    block_large = MatrixProductStateAttention(embed_dim=4, n_log_tokens=4, bond_dim=4)
    n_small = sum(p.numel() for p in block_small.cores)
    n_large = sum(p.numel() for p in block_large.cores)
    # χ goes 2 → 4 quadruples middle-core count, doubles boundary cores.
    assert n_large > 2 * n_small
    assert n_large < 5 * n_small


def test_gradient_flow_through_cores() -> None:
    block = MatrixProductStateAttention(embed_dim=4, n_log_tokens=3, bond_dim=2)
    x = torch.randn(1, 8, 4, requires_grad=True)
    out = block(x)
    out.sum().backward()
    for core in block.cores:
        assert core.grad is not None


def test_truncation_error_bound() -> None:
    """The Schmidt-tail bound returns the sum of squared truncated singular values."""
    sv = torch.tensor([1.0, 0.5, 0.25, 0.1])
    err = MatrixProductStateAttention.truncation_error_bound(sv, bond_dim=2)
    expected = 0.25 ** 2 + 0.1 ** 2
    assert err.item() == pytest.approx(expected, abs=1e-6)


def test_truncation_at_full_rank_is_zero() -> None:
    sv = torch.tensor([1.0, 0.5, 0.25])
    err = MatrixProductStateAttention.truncation_error_bound(sv, bond_dim=3)
    assert err.item() == 0.0


def test_invalid_input_shape() -> None:
    block = MatrixProductStateAttention(embed_dim=4, n_log_tokens=3, bond_dim=2)
    with pytest.raises(ValueError):
        block(torch.randn(2, 8))  # missing token axis
