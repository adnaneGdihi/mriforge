"""Tests for FrozenBackboneAdapter (multi-contrast Item 1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.frozen_backbone_adapter import FrozenBackboneAdapter


def _two_layer_backbone() -> nn.Module:
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))


def _head() -> nn.Module:
    return nn.Conv2d(4, 1, 1)


def test_freeze_backbone_disables_grad() -> None:
    adapter = FrozenBackboneAdapter(_two_layer_backbone(), _head(), freeze=True)
    backbone_params = list(adapter.backbone.parameters())
    head_params = list(adapter.head.parameters())
    assert all(not p.requires_grad for p in backbone_params)
    assert all(p.requires_grad for p in head_params)


def test_trainable_param_count_is_head_only() -> None:
    adapter = FrozenBackboneAdapter(_two_layer_backbone(), _head(), freeze=True)
    expected = sum(p.numel() for p in adapter.head.parameters())
    assert adapter.trainable_parameters() == expected


def test_forward_runs() -> None:
    adapter = FrozenBackboneAdapter(_two_layer_backbone(), _head(), freeze=True)
    x = torch.randn(2, 1, 8, 8)
    out = adapter(x)
    assert out.shape == (2, 1, 8, 8)


def test_freeze_can_be_disabled() -> None:
    adapter = FrozenBackboneAdapter(_two_layer_backbone(), _head(), freeze=False)
    assert all(p.requires_grad for p in adapter.backbone.parameters())


def test_missing_checkpoint_raises() -> None:
    import pytest
    with pytest.raises(FileNotFoundError):
        FrozenBackboneAdapter(
            _two_layer_backbone(),
            _head(),
            checkpoint_uri="/tmp/__never_exists_v6_3.pt",
        )
