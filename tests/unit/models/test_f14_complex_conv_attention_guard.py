"""Regression test for F14 — RE-EVALUATED 2026-05-23.

Original F14 (smoke_audit_20260521 round 7) added an ``__init__``-time
guard in ``KSpaceDownsampleBlock`` / ``KSpaceUpsampleBlock`` that raised
whenever ``use_complex_conv=True`` was combined with any of 7 attention
types, on the theory that a true-complex tensor would hit a real bias and
crash with ``Input type (c10::complex<float>) and bias type (float)``.

That premise was wrong for THIS generator. The motivating crash
(``experiment_12_image_cold_diffusion_v2``) is ``model_type:
image_cold_diffusion`` → ``ImageColdDiffusionUNet``, a *different* backbone
that never uses these blocks. In ``kspace_cold_diffusion_generator`` (and
``complex_unet``) the representation is interleaved-real ``[B, 2C, H, W]``
throughout — ``ComplexConv2d`` (complex_conv.py:194) and
``ComplexActivation`` (activation.py:44) both emit interleaved-real — so
every attention block, including the phase-aware KAN / WaveletFreq novelty
blocks that are the experiment-11 research contribution, forwards cleanly.

The guard was therefore a misattributed false-positive that blocked the
experiment-11 attention cohort (kan_dual_domain headline + the attention
shootout / ablations). It has been removed. This test pins the corrected
behaviour:

1. Every attention type BUILDS with ``use_complex_conv=True`` (no raise).
2. Every attention type FORWARDS cleanly on interleaved-real input,
   returning a real (non-complex) tensor.
3. An unsupported ``attention_type`` still fails loud (CLAUDE.md #9).

See docs/smoke_audit_20260523_fixes.rst.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Real-valued + phase-aware attention names dispatched by the blocks.
# 'spatial' (CBAM) is Up-block only; keep it out of the shared list.
ATTENTION_TYPES = [
    "self",
    "kernelized",
    "sparse",
    "channel",
    "dual_domain",
    "kan_dual_domain",
    "wavelet_freq",
]

IN_CH, OUT_CH, H, W, TDIM = 8, 16, 32, 32, 256


@pytest.fixture
def downsample_cls():
    from mriforge.models.generators.kspace_cold_diffusion_generator import (
        KSpaceDownsampleBlock,
    )
    return KSpaceDownsampleBlock


@pytest.fixture
def upsample_cls():
    from mriforge.models.generators.kspace_cold_diffusion_generator import (
        KSpaceUpsampleBlock,
    )
    return KSpaceUpsampleBlock


@pytest.mark.parametrize("attention_type", ATTENTION_TYPES)
def test_downsample_builds_and_forwards_with_complex_conv(
    downsample_cls, attention_type
) -> None:
    """Complex-conv + any attention BUILDS and FORWARDS on interleaved-real
    input, returning a real tensor (no complex-dtype crash)."""
    blk = downsample_cls(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        attention_type=attention_type,
        use_complex_conv=True,
        time_embedding_dim=TDIM,
        feature_domain="kspace",
    )
    x = torch.randn(2, IN_CH, H, W)
    t = torch.randn(2, TDIM)
    out = blk(x, t)
    out = out[0] if isinstance(out, tuple) else out
    assert not torch.is_complex(out), f"{attention_type}: output must be real"
    assert torch.isfinite(out).all(), f"{attention_type}: non-finite output"


@pytest.mark.parametrize("attention_type", ATTENTION_TYPES + ["spatial"])
def test_upsample_builds_with_complex_conv(upsample_cls, attention_type) -> None:
    """Up block mirrors the down block — building with complex conv +
    any attention must not raise."""
    upsample_cls(
        in_channels=OUT_CH,
        out_channels=IN_CH,
        attention_type=attention_type,
        use_complex_conv=True,
        time_embedding_dim=TDIM,
        feature_domain="kspace",
    )


def test_complex_plus_none_attention_still_allowed(downsample_cls, upsample_cls) -> None:
    """``attention_type='none'`` remains valid with complex conv."""
    downsample_cls(
        in_channels=IN_CH, out_channels=OUT_CH, attention_type="none",
        use_complex_conv=True, time_embedding_dim=TDIM,
        feature_domain="kspace",
    )
    upsample_cls(
        in_channels=OUT_CH, out_channels=IN_CH, attention_type="none",
        use_complex_conv=True, time_embedding_dim=TDIM,
        feature_domain="kspace",
    )


def test_unsupported_attention_type_still_fails_loud(downsample_cls) -> None:
    """The legitimate dispatch guard (CLAUDE.md #9) still rejects an
    attention_type that has no registered handler."""
    with pytest.raises(ValueError, match=r"[Uu]nsupported attention_type"):
        downsample_cls(
            in_channels=IN_CH,
            out_channels=OUT_CH,
            attention_type="not_a_real_attention",
            use_complex_conv=True,
            time_embedding_dim=TDIM,
            feature_domain="kspace",
        )
