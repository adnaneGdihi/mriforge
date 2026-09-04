"""Tests for fusion blocks.

Targets ``spectramr.models.blocks.fusion``. Six fusion strategies for
combining multiple reconstruction outputs:

- ``AttentionFusionBlock``: learned attention weights
- ``FeatureFusionBlock``: channel + spatial attention
- ``MultiScaleFusionBlock``: cross-scale processing (heavier; smoke test)
- ``AdaptiveFusionBlock``: softmax-weighted blending
- ``ResidualFusionBlock``: residual connection preserved
- ``MultiModalFusionBlock``: multi-modal blending (heavier; smoke test)

Plus the ``create_fusion_block`` factory + ``fuse_reconstructions`` helper.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.fusion import (
    AdaptiveFusionBlock,
    AttentionFusionBlock,
    FeatureFusionBlock,
    ResidualFusionBlock,
    create_fusion_block,
    fuse_reconstructions,
)


# ---------------------------------------------------------------------------
# AttentionFusionBlock
# ---------------------------------------------------------------------------


def test_attention_fusion_output_shape() -> None:
    """Output preserves input spatial shape and channel count."""
    block = AttentionFusionBlock(channels=16, num_inputs=2)
    x1 = torch.randn(1, 16, 8, 8)
    x2 = torch.randn(1, 16, 8, 8)
    out = block([x1, x2])
    assert out.shape == (1, 16, 8, 8)


def test_attention_fusion_wrong_num_inputs_raises() -> None:
    """Mismatched input count → ``ValueError``."""
    block = AttentionFusionBlock(channels=8, num_inputs=2)
    x = torch.randn(1, 8, 4, 4)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        block([x])


def test_attention_fusion_handles_low_channel_count() -> None:
    """Channels < reduction_ratio → reduced=1 (no zero-conv)."""
    block = AttentionFusionBlock(channels=4, num_inputs=2, reduction_ratio=16)
    x1 = torch.randn(1, 4, 4, 4)
    x2 = torch.randn(1, 4, 4, 4)
    out = block([x1, x2])
    assert out.shape == (1, 4, 4, 4)


def test_attention_fusion_gradient_flows() -> None:
    """Backward through attention fusion reaches inputs."""
    block = AttentionFusionBlock(channels=8, num_inputs=2)
    x1 = torch.randn(1, 8, 4, 4, requires_grad=True)
    x2 = torch.randn(1, 8, 4, 4, requires_grad=True)
    block([x1, x2]).sum().backward()
    assert x1.grad is not None
    assert x2.grad is not None


# ---------------------------------------------------------------------------
# FeatureFusionBlock
# ---------------------------------------------------------------------------


def test_feature_fusion_output_shape() -> None:
    """Channel attention + spatial attention; output preserves channels."""
    block = FeatureFusionBlock(channels=16, num_inputs=2)
    x1 = torch.randn(1, 16, 8, 8)
    x2 = torch.randn(1, 16, 8, 8)
    out = block([x1, x2])
    assert out.shape == (1, 16, 8, 8)


def test_feature_fusion_wrong_input_count_raises() -> None:
    """Mismatched input count → ``ValueError``."""
    block = FeatureFusionBlock(channels=16, num_inputs=2)
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        block([torch.randn(1, 16, 4, 4)])


# ---------------------------------------------------------------------------
# AdaptiveFusionBlock
# ---------------------------------------------------------------------------


def test_adaptive_fusion_output_shape() -> None:
    """Output preserves channel count after softmax-weighted blend."""
    block = AdaptiveFusionBlock(channels=8, num_inputs=2)
    out = block([torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)])
    assert out.shape == (1, 8, 4, 4)


def test_adaptive_fusion_with_explicit_weights() -> None:
    """Explicit ``weights=[1.0, 0.0]`` → output uses only first input (modulo conv)."""
    block = AdaptiveFusionBlock(channels=4, num_inputs=2)
    x1 = torch.randn(1, 4, 4, 4)
    x2 = torch.randn(1, 4, 4, 4)
    out_a = block([x1, x2], weights=[1.0, 0.0])
    out_b = block([x1, x2], weights=[0.0, 1.0])
    # Different weight configurations → different outputs
    assert not torch.allclose(out_a, out_b)


def test_adaptive_fusion_temperature_persisted() -> None:
    """Temperature attribute is stored."""
    block = AdaptiveFusionBlock(channels=4, num_inputs=2, temperature=2.0)
    assert block.temperature == 2.0


# ---------------------------------------------------------------------------
# ResidualFusionBlock
# ---------------------------------------------------------------------------


def test_residual_fusion_output_shape() -> None:
    """Output shape preserved."""
    block = ResidualFusionBlock(channels=8, num_inputs=2)
    out = block([torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)])
    assert out.shape == (1, 8, 4, 4)


def test_residual_fusion_weights_are_learnable() -> None:
    """``fusion_weights`` is a registered parameter."""
    block = ResidualFusionBlock(channels=4, num_inputs=3)
    assert block.fusion_weights.requires_grad is True
    assert block.fusion_weights.shape == (3,)


# ---------------------------------------------------------------------------
# create_fusion_block factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fusion_type, expected_class",
    [
        ("attention", AttentionFusionBlock),
        ("feature", FeatureFusionBlock),
        ("adaptive", AdaptiveFusionBlock),
        ("residual", ResidualFusionBlock),
    ],
)
def test_factory_returns_correct_class(fusion_type: str, expected_class) -> None:
    """Factory returns the right class for each fusion type."""
    block = create_fusion_block(fusion_type, channels=8, num_inputs=2)
    assert isinstance(block, expected_class)


def test_factory_case_insensitive() -> None:
    """Factory is case-insensitive."""
    block = create_fusion_block("ATTENTION", channels=8, num_inputs=2)
    assert isinstance(block, AttentionFusionBlock)


def test_factory_unknown_type_raises() -> None:
    """Unknown fusion_type → ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown fusion type"):
        create_fusion_block("not_a_fusion", channels=8, num_inputs=2)


# ---------------------------------------------------------------------------
# fuse_reconstructions helper
# ---------------------------------------------------------------------------


def test_fuse_reconstructions_helper() -> None:
    """End-to-end fusion via the helper."""
    recon1 = torch.randn(1, 8, 16, 16)
    recon2 = torch.randn(1, 8, 16, 16)
    out = fuse_reconstructions([recon1, recon2], fusion_type="attention")
    assert out.shape == recon1.shape


def test_fuse_reconstructions_requires_at_least_two() -> None:
    """Less than 2 inputs → ``ValueError``."""
    with pytest.raises(ValueError, match="at least 2 reconstructions"):
        fuse_reconstructions([torch.randn(1, 8, 4, 4)])
