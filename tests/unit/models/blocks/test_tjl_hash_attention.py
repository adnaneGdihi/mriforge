"""Tests for TJLHashAttention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.tjl_hash_attention import (
    TJLHashAttention,
    tensorised_jl_hash,
)


def test_tjl_hash_preserves_distance_approximately() -> None:
    """JL guarantee: ratio (||φ(x) − φ(y)||₂ / ||x − y||_F) → 1 in expectation.

    For a Kronecker-structured projection :math:`R_1\\otimes R_2` of order 2
    the expected squared-norm preservation is exact; the empirical ratio
    converges to 1 at rate :math:`O(1/\\sqrt m)`. We test the *averaged*
    ratio over many pairs, which is what the theorem actually states.
    """
    torch.manual_seed(0)
    n_pairs = 256
    m = 16
    signals = torch.randn(1, n_pairs, 8, 4)
    proj1 = torch.randn(m, 8) / (m ** 0.5)
    proj2 = torch.randn(m, 4) / (m ** 0.5)
    hashes = tensorised_jl_hash(signals, proj1, proj2)
    # All pairwise (i, i+1) ratios.
    ratios = []
    for i in range(n_pairs - 1):
        a, b = hashes[0, i], hashes[0, i + 1]
        a_sig, b_sig = signals[0, i], signals[0, i + 1]
        denom = (a_sig - b_sig).norm()
        if denom > 1e-6:
            ratios.append(((a - b).norm() / denom).item())
    mean_ratio = sum(ratios) / len(ratios)
    assert 0.7 < mean_ratio < 1.5


def test_forward_shape_preserved() -> None:
    block = TJLHashAttention(
        time_dim=8, coil_dim=4, hash_dim=4, value_dim=16
    )
    signals = torch.randn(2, 6, 8, 4)
    out = block(signals)
    assert out.shape == (2, 6, 16)


def test_projection_matrices_are_buffers() -> None:
    """Projections are frozen (not learnable) — JL theorem hypothesis."""
    block = TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=4)
    state = dict(block.named_buffers())
    assert "proj1" in state
    assert "proj2" in state
    assert state["proj1"].requires_grad is False


def test_seed_reproducibility() -> None:
    block_a = TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=4, seed=42)
    block_b = TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=4, seed=42)
    assert torch.allclose(block_a.proj1, block_b.proj1)
    assert torch.allclose(block_a.proj2, block_b.proj2)


def test_gradient_flow() -> None:
    block = TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=4)
    signals = torch.randn(1, 4, 4, 2, requires_grad=True)
    out = block(signals)
    out.sum().backward()
    assert signals.grad is not None


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=1)
    block = TJLHashAttention(time_dim=4, coil_dim=2, hash_dim=4)
    with pytest.raises(ValueError):
        block(torch.randn(1, 4))
    with pytest.raises(ValueError):
        block(torch.randn(1, 4, 5, 2))  # wrong time_dim
    with pytest.raises(ValueError):
        block(torch.randn(1, 4, 4, 3))  # wrong coil_dim


def test_tensorised_hash_invalid_shapes() -> None:
    bad_sig = torch.randn(4, 8)
    proj1 = torch.randn(4, 8)
    proj2 = torch.randn(4, 4)
    with pytest.raises(ValueError):
        tensorised_jl_hash(bad_sig, proj1, proj2)
    sig = torch.randn(1, 2, 8, 4)
    with pytest.raises(ValueError):
        tensorised_jl_hash(sig, torch.randn(4, 5), proj2)
