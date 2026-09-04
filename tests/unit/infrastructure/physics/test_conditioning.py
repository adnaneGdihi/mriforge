"""Tests for ``PhysicsInformedConditioning``.

Targets ``spectramr.infrastructure.physics.conditioning``. AdaGN-style FiLM
modulation of feature maps by a physics embedding (acceleration
factor, timestep, noise level, etc.).

Categories:

- Unit: 4D input shape preserved; broadcasting works for 5D too
- Property: zero scale + zero shift → output equals input
- Property: gradient flows to both x and emb
- Sanity-shape: parametrised over channel/spatial dims
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.conditioning import PhysicsInformedConditioning


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_persists_dimensions() -> None:
    """Sequence has SiLU + Linear; Linear has channel_dim*2 outputs."""
    cond = PhysicsInformedConditioning(embedding_dim=16, channel_dim=8)
    assert isinstance(cond.fc, torch.nn.Sequential)
    # Last Linear should map 16 → 16 (8 channels × 2)
    last_linear = cond.fc[-1]
    assert last_linear.in_features == 16
    assert last_linear.out_features == 16


# ---------------------------------------------------------------------------
# Forward — 4D
# ---------------------------------------------------------------------------


def test_forward_4d_preserves_shape() -> None:
    """``[B, C, H, W]`` input → output of identical shape."""
    cond = PhysicsInformedConditioning(embedding_dim=16, channel_dim=8)
    x = torch.randn(2, 8, 16, 16)
    emb = torch.randn(2, 16)
    out = cond(x, emb)
    assert out.shape == x.shape


def test_forward_5d_preserves_shape() -> None:
    """``[B, C, D, H, W]`` input also preserves shape (broadcasting handles it)."""
    cond = PhysicsInformedConditioning(embedding_dim=8, channel_dim=4)
    x = torch.randn(1, 4, 4, 8, 8)
    emb = torch.randn(1, 8)
    out = cond(x, emb)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Identity-like behaviour
# ---------------------------------------------------------------------------


def test_zero_emb_with_zero_init_yields_input() -> None:
    """With Linear weights & bias zero-initialised, output equals input."""
    cond = PhysicsInformedConditioning(embedding_dim=16, channel_dim=8)
    last_linear = cond.fc[-1]
    with torch.no_grad():
        last_linear.weight.zero_()
        last_linear.bias.zero_()
    x = torch.randn(1, 8, 4, 4)
    emb = torch.randn(1, 16)
    # scale=0, shift=0 → x * 1 + 0 = x
    out = cond(x, emb)
    assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_input_and_embedding() -> None:
    """Backward through conditioning reaches both ``x`` and ``emb``."""
    cond = PhysicsInformedConditioning(embedding_dim=16, channel_dim=8)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    emb = torch.randn(1, 16, requires_grad=True)
    cond(x, emb).sum().backward()
    assert x.grad is not None
    assert emb.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(emb.grad).all()


# ---------------------------------------------------------------------------
# Sanity-shape matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels, h, w, emb_dim",
    [(4, 8, 8, 8), (16, 16, 16, 32), (32, 32, 32, 64)],
    ids=["c4_e8", "c16_e32", "c32_e64"],
)
def test_conditioning_shape_matrix(
    channels: int, h: int, w: int, emb_dim: int
) -> None:
    """Conditioning preserves shape across channel / embedding configurations."""
    cond = PhysicsInformedConditioning(embedding_dim=emb_dim, channel_dim=channels)
    x = torch.randn(1, channels, h, w)
    emb = torch.randn(1, emb_dim)
    out = cond(x, emb)
    assert out.shape == (1, channels, h, w)
