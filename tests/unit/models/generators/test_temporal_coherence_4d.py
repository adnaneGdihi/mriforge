"""Regression test for F11 (smoke audit 2026-06-04).

``temporal_coherence_4d``'s ``_TemporalAttention`` computed attention over the
flattened SPATIAL axis: ``q.transpose(-1,-2) @ k`` built a ``[B, H*W, H*W]``
matrix. On a 256² grid that is a 65536×65536 ≈ 64 GiB allocation that OOM'd at
iteration 1 (exp_vf_07, dispatch 20260603_200125 task 29). It must be
channel/temporal attention — ``[B, C, C]`` — contracting over the HW feature
axis. This test forwards at 256² (which the buggy spatial version cannot
allocate) and pins the output shape + residual + finiteness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.temporal_coherence_4d import (  # noqa: E402
    _TemporalAttention,
)


def test_temporal_attention_is_channel_not_spatial_no_oom() -> None:
    attn = _TemporalAttention(ch=8, temporal_dim=8).eval()
    # HW = 65536; the buggy [HW, HW] attention would need ~16 GiB even at B=1.
    x = torch.randn(1, 8, 256, 256)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_temporal_attention_preserves_shape_small() -> None:
    attn = _TemporalAttention(ch=4, temporal_dim=4).eval()
    x = torch.randn(2, 4, 16, 16)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (2, 4, 16, 16)
    # Residual connection: identity-ish when proj weights are small/zeroed.
    with torch.no_grad():
        attn.proj.weight.zero_()
        attn.proj.bias.zero_()
        out0 = attn(x)
    assert torch.allclose(out0, x)


def test_attention_cost_is_quadratic_in_channels_not_pixels() -> None:
    """White-box: doubling the spatial grid must NOT blow up memory/time the way
    a [HW, HW] attention would. We assert it runs at a grid size where the old
    op would allocate >1 GiB (128² → [16384, 16384] ≈ 1 GiB per batch elem)."""
    attn = _TemporalAttention(ch=8, temporal_dim=8).eval()
    with torch.no_grad():
        out = attn(torch.randn(1, 8, 128, 128))
    assert out.shape == (1, 8, 128, 128)
