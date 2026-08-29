"""Smoke tests for the 5 breakthrough attention generators (P6–P10).

Each generator wraps one of the new attention blocks in
:mod:`mriforge.models.blocks` and is registered via ``@register_model`` so the
audit-ladder schema can resolve it by name. The tests verify:

1. The model is reachable through ``MODEL_REGISTRY`` by its registered name.
2. A forward pass produces output of the expected shape.
3. Gradients flow through the block.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.generators.breakthrough_attention_generators import (
    BlochLRSAttentionUNet,
    LanczosAttentionUNet,
    MPSAttentionUNet,
    TJLMRFEstimator,
    ToeplitzAttentionUNet,
)
from mriforge.models.registry import MODEL_REGISTRY

_RECON_GENERATORS = [
    ("toeplitz_attention_unet", ToeplitzAttentionUNet),
    ("bloch_lrs_attention_unet", BlochLRSAttentionUNet),
    ("lanczos_attention_unet", LanczosAttentionUNet),
    ("mps_attention_unet", MPSAttentionUNet),
    ("tjl_mrf_estimator", TJLMRFEstimator),
]


@pytest.mark.parametrize("name,cls", _RECON_GENERATORS)
def test_generator_is_registered_by_name(name: str, cls: type) -> None:
    entry = MODEL_REGISTRY.get(name)
    assert entry is not None, f"{name} not in MODEL_REGISTRY"
    assert entry["class"] is cls
    assert entry["mode"] == "reconstruction"


@pytest.mark.parametrize("name,cls", _RECON_GENERATORS)
def test_generator_forward_shape_preserved(name: str, cls: type) -> None:
    model = cls()
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64), f"{name} produced {tuple(out.shape)}"


@pytest.mark.parametrize("name,cls", _RECON_GENERATORS)
def test_generator_gradient_flows(name: str, cls: type) -> None:
    model = cls()
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    # At least one trainable parameter should have a gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_bloch_lrs_generator_accepts_bloch_dictionary() -> None:
    """The Bloch-LRS variant carries through the optional dictionary buffer."""
    rank = 4
    bloch_dict = torch.randn(rank, rank)
    model = BlochLRSAttentionUNet(rank=rank, bloch_dictionary=bloch_dict)
    assert model.attn.bloch_basis is not None
    assert model.attn.bloch_basis.shape == (rank, rank)


def test_mps_generator_bond_dim_propagates() -> None:
    model = MPSAttentionUNet(bond_dim=8)
    assert model.attn.bond_dim == 8


def test_tjl_mrf_estimator_accepts_mrf_signal_shape() -> None:
    """The TJL regressor accepts (B, N, T, C) inputs directly (non-image path)."""
    model = TJLMRFEstimator(time_dim=4, coil_dim=2, hash_dim=4, value_dim=8)
    sig = torch.randn(2, 16, 4, 2)
    out = model(sig)
    assert out.shape == (2, 16, model.out_channels)
