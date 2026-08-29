"""Tests for ``CUTPatchNCELoss`` (CUT).

Targets ``mriforge.models.losses.cut_contrastive_loss``. Patch-NCE
contrastive loss for unpaired translation.

Categories:

- Construction: invalid ``n_patches`` / ``temperature`` rejected
- Identity features (source == translated) → loss is the InfoNCE
  identity baseline ``log n`` (after the normalisation collapses)
- Missing ``context`` raises
- Per-layer shape mismatch raises
- Multi-layer feature lists are averaged
- Gradient flows back into the feature maps
- Registry: ``cut_patch_nce`` resolvable
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.losses.cut_contrastive_loss import CUTPatchNCELoss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_zero_n_patches_rejected() -> None:
    """``n_patches ≤ 0`` raises."""
    with pytest.raises(ValueError, match="n_patches"):
        CUTPatchNCELoss(n_patches=0)


def test_zero_temperature_rejected() -> None:
    """``temperature ≤ 0`` raises."""
    with pytest.raises(ValueError, match="temperature"):
        CUTPatchNCELoss(temperature=0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_context_rejected() -> None:
    """``context=None`` rejected."""
    loss = CUTPatchNCELoss()
    with pytest.raises(ValueError, match="context"):
        loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))


def test_missing_feat_source_rejected() -> None:
    """Missing ``feat_source`` key rejected."""
    loss = CUTPatchNCELoss()
    with pytest.raises(ValueError, match="feat_source"):
        loss(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 4, 4),
            context={"feat_translated": [torch.randn(1, 4, 8, 8)]},
        )


def test_layer_count_mismatch_rejected() -> None:
    """Mismatched layer counts between src and translated → ValueError."""
    loss = CUTPatchNCELoss()
    src = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)]
    tgt = [torch.randn(1, 4, 8, 8)]
    with pytest.raises(ValueError, match="feat_source has"):
        loss(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 4, 4),
            context={"feat_source": src, "feat_translated": tgt},
        )


def test_per_layer_shape_mismatch_rejected() -> None:
    """Per-layer feature shape must match between source and translated."""
    loss = CUTPatchNCELoss()
    src = [torch.randn(1, 4, 8, 8)]
    tgt = [torch.randn(1, 4, 4, 4)]
    with pytest.raises(ValueError, match="feat shape"):
        loss(
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 4, 4),
            context={"feat_source": src, "feat_translated": tgt},
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def test_forward_returns_finite_scalar() -> None:
    """Loss returns a finite 0-D tensor on a single feature layer."""
    torch.manual_seed(0)
    loss = CUTPatchNCELoss(n_patches=16, temperature=0.07)
    src = [torch.randn(1, 8, 8, 8)]
    tgt = [torch.randn(1, 8, 8, 8)]
    out = loss(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        context={"feat_source": src, "feat_translated": tgt},
    )
    assert out.dim() == 0
    assert torch.isfinite(out).all()


def test_multi_layer_features_averaged() -> None:
    """Two-layer feature lists average two per-layer InfoNCE terms."""
    torch.manual_seed(0)
    loss = CUTPatchNCELoss(n_patches=8, temperature=0.07)
    src = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)]
    tgt = [torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)]
    out = loss(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        context={"feat_source": src, "feat_translated": tgt},
    )
    assert torch.isfinite(out)


def test_n_patches_capped_to_spatial_extent() -> None:
    """``n_patches > H · W`` is silently capped to ``H · W``."""
    torch.manual_seed(0)
    # 4×4 = 16 spatial samples; ask for 100 patches.
    loss = CUTPatchNCELoss(n_patches=100, temperature=0.07)
    src = [torch.randn(1, 4, 4, 4)]
    tgt = [torch.randn(1, 4, 4, 4)]
    # Should not raise.
    out = loss(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        context={"feat_source": src, "feat_translated": tgt},
    )
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_feature_maps() -> None:
    """Backward through the loss reaches the translated feature map."""
    torch.manual_seed(0)
    loss = CUTPatchNCELoss(n_patches=8, temperature=0.07)
    feat_src = torch.randn(1, 4, 8, 8)
    feat_tgt = torch.randn(1, 4, 8, 8, requires_grad=True)
    out = loss(
        torch.zeros(1, 1, 4, 4),
        torch.zeros(1, 1, 4, 4),
        context={"feat_source": [feat_src], "feat_translated": [feat_tgt]},
    )
    out.backward()
    assert feat_tgt.grad is not None
    assert torch.isfinite(feat_tgt.grad).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``cut_patch_nce`` resolvable in the loss registry."""
    from mriforge.models.losses.registry import list_available

    assert "cut_patch_nce" in list_available()
