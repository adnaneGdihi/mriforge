"""Tests for SparseAutoencoder (PR-24 / Part-I F)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.sparse_autoencoder import SparseAutoencoder


def test_encode_decode_shape() -> None:
    sae = SparseAutoencoder(input_dim=16, expansion_factor=4)
    x = torch.randn(4, 16)
    x_hat, z = sae(x)
    assert x_hat.shape == (4, 16)
    assert z.shape == (4, 64)


def test_top_k_sparsity_keeps_only_k_active() -> None:
    sae = SparseAutoencoder(input_dim=8, expansion_factor=4, top_k=3)
    x = torch.randn(2, 8)
    z = sae.encode(x)
    n_active = (z > 0).float().sum(dim=-1)
    # top_k acts after ReLU, so the count of nonzeros is at most top_k.
    assert (n_active <= 3).all()


def test_loss_returns_recon_and_sparsity_terms() -> None:
    sae = SparseAutoencoder(input_dim=8, expansion_factor=4, l1_lambda=1e-3)
    x = torch.randn(2, 8)
    total, parts = sae.loss(x)
    assert total.dim() == 0
    assert "loss_recon" in parts
    assert "loss_sparsity" in parts
    assert "active_fraction" in parts
    assert 0.0 <= parts["active_fraction"].item() <= 1.0


def test_l1_lambda_increases_sparsity() -> None:
    """Higher l1_lambda → more zero activations after one optimisation step."""
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    low = SparseAutoencoder(input_dim=8, expansion_factor=4, l1_lambda=1e-6)
    high = SparseAutoencoder(input_dim=8, expansion_factor=4, l1_lambda=1.0)
    # Same init.
    high.load_state_dict(low.state_dict())

    for sae in (low, high):
        opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
        for _ in range(20):
            opt.zero_grad()
            t, _ = sae.loss(x)
            t.backward()
            opt.step()
    _, low_parts = low.loss(x)
    _, high_parts = high.loss(x)
    assert high_parts["active_fraction"] <= low_parts["active_fraction"] + 1e-3
