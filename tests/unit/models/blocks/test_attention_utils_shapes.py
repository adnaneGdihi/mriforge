"""Sanity-shape + canary tests for ``attention_utils``.

Targets ``EfficientAttentionLayer`` (``nn.Module``) and the
``AttentionWrapper.compute_attention`` forward path — the parts that
were NOT covered by the existing ``test_attention_utils.py`` which only
exercised config and construction.

Forward contract
----------------
- ``EfficientAttentionLayer(embed_dim, num_heads)``:
    input  ``(B, L, embed_dim)``
    output ``(B, L, embed_dim)``
- ``AttentionWrapper.compute_attention(q, k, v)``:
    input  each ``(B, H, L, head_dim)``
    output ``(B, H, L, head_dim)``
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.attention_utils import (
    AttentionBackend,
    AttentionConfig,
    AttentionWrapper,
    EfficientAttentionLayer,
)

# ── EfficientAttentionLayer canary ──────────────────────────────────────


class TestEfficientAttentionLayerCanary:
    def test_canary_minimal_forward_is_finite(self) -> None:
        """Minimal construction + forward → finite output, same shape."""
        layer = EfficientAttentionLayer(embed_dim=16, num_heads=4)
        layer.eval()
        x = torch.randn(2, 8, 16)
        out = layer(x)
        assert out.shape == (2, 8, 16)
        assert torch.isfinite(out).all()

    def test_canary_batch1_seq1(self) -> None:
        """Edge-case: batch=1, seq=1."""
        layer = EfficientAttentionLayer(embed_dim=8, num_heads=2)
        layer.eval()
        x = torch.randn(1, 1, 8)
        out = layer(x)
        assert out.shape == (1, 1, 8)
        assert torch.isfinite(out).all()

    def test_canary_output_channels_match(self) -> None:
        """Output embed_dim always equals input embed_dim."""
        for embed_dim, num_heads in [(16, 4), (32, 8), (64, 8)]:
            layer = EfficientAttentionLayer(embed_dim=embed_dim, num_heads=num_heads)
            layer.eval()
            x = torch.randn(1, 4, embed_dim)
            out = layer(x)
            assert out.shape[-1] == embed_dim, (
                f"embed_dim={embed_dim}: got output dim {out.shape[-1]}"
            )


# ── EfficientAttentionLayer shape matrix ────────────────────────────────

# (batch, seq_len, embed_dim, num_heads)
_SHAPE_CASES = [
    (1, 4, 8, 2),
    (2, 16, 16, 4),
    (1, 32, 32, 8),
    (3, 8, 64, 8),
]


@pytest.mark.parametrize("B,L,E,H", _SHAPE_CASES, ids=[
    f"B{b}L{l}E{e}H{h}" for b, l, e, h in _SHAPE_CASES
])
def test_efficient_attention_layer_forward_shape(B: int, L: int, E: int, H: int) -> None:
    """Forward returns ``(B, L, E)`` for all matrix entries."""
    layer = EfficientAttentionLayer(embed_dim=E, num_heads=H)
    layer.eval()
    x = torch.randn(B, L, E)
    out = layer(x)
    assert out.shape == (B, L, E)
    assert torch.isfinite(out).all()


def test_efficient_attention_layer_wrong_embed_dim_raises() -> None:
    """Passing a tensor with the wrong embed_dim triggers an error."""
    layer = EfficientAttentionLayer(embed_dim=16, num_heads=4)
    layer.eval()
    # (B, L, wrong_C)
    x = torch.randn(1, 8, 32)
    with pytest.raises((RuntimeError, Exception)):
        layer(x)


# ── AttentionWrapper.compute_attention ──────────────────────────────────


class TestAttentionWrapperForward:
    def test_pytorch_backend_forward(self) -> None:
        """Pytorch backend compute_attention produces correct shape + finite."""
        cfg = AttentionConfig(backend=AttentionBackend.PYTORCH)
        wrapper = AttentionWrapper(cfg)
        B, H, L, D = 2, 4, 8, 16
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        out = wrapper.compute_attention(q, k, v)
        assert out.shape == (B, H, L, D)
        assert torch.isfinite(out).all()

    def test_memory_efficient_backend_forward(self) -> None:
        """memory_efficient backend returns same shape."""
        cfg = AttentionConfig(backend=AttentionBackend.MEMORY_EFFICIENT)
        wrapper = AttentionWrapper(cfg)
        B, H, L, D = 1, 2, 4, 8
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        out = wrapper.compute_attention(q, k, v)
        assert out.shape == (B, H, L, D)
        assert torch.isfinite(out).all()

    def test_wrong_ndim_raises(self) -> None:
        """3-D query raises ``RuntimeError``."""
        wrapper = AttentionWrapper()
        q = torch.randn(2, 8, 16)  # 3-D — must be 4-D
        k = torch.randn(2, 8, 16)
        v = torch.randn(2, 8, 16)
        with pytest.raises(RuntimeError, match="dimensions"):
            wrapper.compute_attention(q, k, v)

    def test_batch_size_mismatch_raises(self) -> None:
        """Mismatched batch sizes raise ``RuntimeError``."""
        wrapper = AttentionWrapper()
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(3, 4, 8, 16)  # different B
        v = torch.randn(2, 4, 8, 16)
        with pytest.raises(RuntimeError, match="atch"):
            wrapper.compute_attention(q, k, v)

    def test_head_mismatch_raises(self) -> None:
        """Mismatched head counts raise ``RuntimeError``."""
        wrapper = AttentionWrapper()
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 8, 8, 16)  # different H
        v = torch.randn(2, 4, 8, 16)
        with pytest.raises(RuntimeError):
            wrapper.compute_attention(q, k, v)

    def test_causal_mask_finite(self) -> None:
        """Causal mode produces finite output."""
        cfg = AttentionConfig(causal=True)
        wrapper = AttentionWrapper(cfg)
        B, H, L, D = 1, 2, 6, 8
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        out = wrapper.compute_attention(q, k, v)
        assert torch.isfinite(out).all()
        assert out.shape == (B, H, L, D)
