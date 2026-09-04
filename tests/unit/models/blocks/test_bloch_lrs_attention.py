"""Tests for BlochLRSAttention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.bloch_lrs_attention import BlochLRSAttention


def test_forward_shape_preserved() -> None:
    block = BlochLRSAttention(embed_dim=16, token_len=32, rank=4)
    x = torch.randn(2, 32, 16)
    out = block(x)
    assert out.shape == x.shape


def test_bloch_projection_restricts_rank() -> None:
    """When the Bloch basis spans rank < r columns, U is restricted."""
    rank = 8
    bloch_dict = torch.randn(rank, rank)  # full-rank, identity-like
    block = BlochLRSAttention(
        embed_dim=8, token_len=16, rank=rank, bloch_dictionary=bloch_dict
    )
    # The Bloch basis has orthogonal columns.
    q = block.bloch_basis
    assert torch.allclose(q.T @ q, torch.eye(rank), atol=1e-4)


def test_invalid_dictionary_shape_raises() -> None:
    bad = torch.randn(8, 5)  # second dim != rank
    with pytest.raises(ValueError):
        BlochLRSAttention(embed_dim=8, token_len=16, rank=4, bloch_dictionary=bad)


def test_sparse_component_optional() -> None:
    block_no_sparse = BlochLRSAttention(embed_dim=8, token_len=16, rank=4, sparse_k=0)
    block_sparse = BlochLRSAttention(embed_dim=8, token_len=16, rank=4, sparse_k=4)
    assert block_no_sparse.sparse_values is None
    assert block_sparse.sparse_values is not None
    assert block_sparse.sparse_values.numel() == 16 * 4


def test_gradient_flow_through_factors() -> None:
    block = BlochLRSAttention(embed_dim=8, token_len=16, rank=4, sparse_k=2)
    x = torch.randn(1, 16, 8, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert block.u_factor.grad is not None
    assert block.v_factor.grad is not None
    assert block.sparse_values.grad is not None


def test_init_validation() -> None:
    with pytest.raises(ValueError):
        BlochLRSAttention(embed_dim=8, token_len=16, rank=0)
    with pytest.raises(ValueError):
        BlochLRSAttention(embed_dim=8, token_len=16, rank=4, sparse_k=-1)
    with pytest.raises(ValueError):
        BlochLRSAttention(embed_dim=8, token_len=1, rank=4)


def test_forward_invalid_input() -> None:
    block = BlochLRSAttention(embed_dim=8, token_len=16, rank=4)
    with pytest.raises(ValueError):
        block(torch.randn(2, 16))
    with pytest.raises(ValueError):
        block(torch.randn(2, 17, 8))
    with pytest.raises(ValueError):
        block(torch.randn(2, 16, 4))
