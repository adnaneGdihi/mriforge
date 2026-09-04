"""Tests for KrylovLanczosAttention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.lanczos_attention import KrylovLanczosAttention


def test_forward_shape_preserved() -> None:
    block = KrylovLanczosAttention(embed_dim=16, num_heads=4, krylov_dim=4)
    x = torch.randn(2, 32, 16)
    out = block(x)
    assert out.shape == x.shape


def test_gradient_flow_through_projections() -> None:
    block = KrylovLanczosAttention(embed_dim=8, num_heads=2, krylov_dim=4)
    x = torch.randn(1, 16, 8, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None
    assert block.q_proj.weight.grad is not None


def test_krylov_dim_validation() -> None:
    with pytest.raises(ValueError):
        KrylovLanczosAttention(embed_dim=16, num_heads=4, krylov_dim=0)


def test_embed_divisibility_validation() -> None:
    with pytest.raises(ValueError):
        KrylovLanczosAttention(embed_dim=15, num_heads=4, krylov_dim=4)


def test_invalid_input_shape_raises() -> None:
    block = KrylovLanczosAttention(embed_dim=8, num_heads=2, krylov_dim=4)
    with pytest.raises(ValueError):
        block(torch.randn(2, 16))
    with pytest.raises(ValueError):
        block(torch.randn(2, 16, 4))
    with pytest.raises(ValueError):
        block(torch.randn(2, 1, 8))


def test_krylov_basis_orthogonality_at_low_k() -> None:
    """For k <= 4, full reorthogonalisation keeps basis vectors near-orthogonal."""
    block = KrylovLanczosAttention(embed_dim=8, num_heads=1, krylov_dim=4)
    x = torch.randn(1, 16, 8)
    q = block.q_proj(x).view(1, 16, 1, 8).transpose(1, 2)
    k = block.k_proj(x).view(1, 16, 1, 8).transpose(1, 2)
    v = block.v_proj(x).view(1, 16, 1, 8).transpose(1, 2)

    def h_apply(u: torch.Tensor) -> torch.Tensor:
        scale = 8 ** -0.5
        kt_u = torch.einsum("bhnd,bhn->bhd", k, u)
        qkt_u = torch.einsum("bhnd,bhd->bhn", q, kt_u)
        qt_u = torch.einsum("bhnd,bhn->bhd", q, u)
        kqt_u = torch.einsum("bhnd,bhd->bhn", k, qt_u)
        return scale * (qkt_u + kqt_u)

    v0 = v.mean(dim=-1)
    v_basis, _, _ = block._lanczos(h_apply, v0)
    # Inner products between distinct basis vectors should be near zero.
    gram = torch.einsum("bhin,bhjn->bhij", v_basis, v_basis)
    off_diag = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    assert off_diag.abs().max().item() < 1e-3
