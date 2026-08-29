"""Regression tests for mriforge.models.transformer.components.

Covers the unwired-knob fix: StandardViT / StandardViTWithKAN advertise an
``out_channels`` kwarg but the decoder head and forward reshape are hardcoded to
a single output channel. Per CLAUDE.md non-negotiable #8/#15 (every exposed knob
must be wired or raise), passing ``out_channels != 1`` must fail loud at
construction rather than silently produce a 1-channel output.
"""

import pytest
import torch

from mriforge.models.transformer.components import StandardViT, StandardViTWithKAN


@pytest.mark.parametrize("cls", [StandardViT, StandardViTWithKAN])
def test_out_channels_not_one_raises(cls):
    """out_channels != 1 is unwired and must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="out_channels=1"):
        cls(
            img_size=32,
            patch_size=16,
            in_channels=1,
            out_channels=2,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )


@pytest.mark.parametrize("cls", [StandardViT, StandardViTWithKAN])
def test_out_channels_one_constructs_and_stores(cls):
    """The supported out_channels=1 path constructs and records the knob."""
    model = cls(
        img_size=32,
        patch_size=16,
        in_channels=1,
        out_channels=1,
        embed_dim=32,
        depth=1,
        num_heads=2,
    )
    assert model.out_channels == 1


def test_standard_vit_forward_single_channel():
    """Positive regression: forward yields a single-channel output of input HxW."""
    torch.manual_seed(0)
    model = StandardViT(
        img_size=32,
        patch_size=16,
        in_channels=1,
        out_channels=1,
        embed_dim=32,
        depth=1,
        num_heads=2,
    ).to("cpu")
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32)


def test_standard_vit_with_kan_forward_single_channel():
    """Positive regression for the KAN variant's single-channel forward."""
    torch.manual_seed(0)
    model = StandardViTWithKAN(
        img_size=32,
        patch_size=16,
        in_channels=1,
        out_channels=1,
        embed_dim=32,
        depth=1,
        num_heads=2,
    ).to("cpu")
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 32, 32)
