"""Tests for the two-level VQ-VAE-2 (``hierarchical_vq_vae``).

Write-only: executed on the cluster, not here.
"""

from __future__ import annotations

import torch

from mriforge.models.vq.hierarchical_vq_vae import HierarchicalVQVAE

from ._vae_base import assert_gradient_flows, assert_recon_shape, assert_registered


def _model() -> HierarchicalVQVAE:
    return HierarchicalVQVAE(
        in_channels=1,
        out_channels=1,
        codebook_size_top=64,
        codebook_size_bot=64,
        embed_dim=16,
        hidden=32,
    )


def test_registry_resolution():
    assert_registered("hierarchical_vq_vae", "vqvae")


def test_constructs_with_no_kwargs():
    HierarchicalVQVAE()


def test_forward_returns_dict_with_finite_vq_loss():
    model = _model()
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert set(out) >= {"x_hat", "vq_loss", "idx_top", "idx_bot"}
    assert_recon_shape(out["x_hat"], x)
    assert torch.isfinite(out["vq_loss"]).all()


def test_codebook_indices_in_range():
    model = _model()
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert int(out["idx_top"].min()) >= 0
    assert int(out["idx_top"].max()) < 64
    assert int(out["idx_bot"].min()) >= 0
    assert int(out["idx_bot"].max()) < 64


def test_deterministic_indices_in_eval():
    """In eval mode (no EMA update) indices repeat on the same input."""
    model = _model().eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1["idx_top"], out2["idx_top"])
    assert torch.equal(out1["idx_bot"], out2["idx_bot"])


def test_gradient_flow():
    model = _model()
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out["x_hat"], x) + out["vq_loss"]
    assert_gradient_flows(model, loss)
