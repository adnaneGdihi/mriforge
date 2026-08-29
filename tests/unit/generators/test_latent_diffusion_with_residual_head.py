"""Tests for the latent-diffusion + pixel-space residual-head wrapper.

Multi-contrast Item 2 (``TODO/backlog_multi_contrast_remaining.md``).
The wrapper composes ``LatentDiffusionGenerator`` (backbone) with a
small UNet residual head that adds back the high-frequency content the
autoencoder bottleneck discarded.
"""
from __future__ import annotations

import pytest
import torch


def test_wrapper_is_registered() -> None:
    """The model must register under its canonical key."""
    import mriforge.models.generators  # noqa: F401
    from mriforge.models.registry import list_models

    assert "latent_diffusion_residual_head" in list_models()


def test_wrapper_supports_contrast_conditioning_capability() -> None:
    """Registry must reflect ``supports_contrast_conditioning=True``."""
    import mriforge.models.generators  # noqa: F401
    from mriforge.models.registry import model_supports

    assert model_supports(
        "latent_diffusion_residual_head", "supports_contrast_conditioning"
    )


def test_residual_head_param_ratio_under_10_percent() -> None:
    """Acceptance criterion: head must stay lightweight (< 10 % of backbone)."""
    from mriforge.models.generators.latent_diffusion_with_residual_head import (
        LatentDiffusionWithResidualHead,
    )

    mdl = LatentDiffusionWithResidualHead(
        backbone={
            "base_channels": 32,
            "timesteps": 10,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
        },
        residual_head={"base_channels": 32, "depth": 3, "contrast_conditioning": True},
        num_contrasts=4,
    )
    ratio = mdl.residual_head_param_ratio()
    assert ratio < 0.10, (
        f"Residual head must be < 10 % of backbone parameters "
        f"(got {ratio:.2%}); the spec calls for a lightweight head."
    )


def test_residual_head_is_initialised_as_identity() -> None:
    """Zero-init of the head's output conv means the wrapper starts as backbone-only.

    This avoids destabilising a pretrained backbone at the moment the
    head is attached -- the head only contributes once it has trained.
    """
    from mriforge.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(in_channels=1, out_channels=1, base_channels=8, depth=2)
    x = torch.randn(1, 1, 16, 16)
    out = head(x)
    # Output conv weights are zero -> the residual must be all-zero at init.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_residual_head_unet_forward_shape() -> None:
    """The head must preserve the input spatial shape."""
    from mriforge.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(in_channels=1, out_channels=1, base_channels=8, depth=2)
    x = torch.randn(2, 1, 16, 16)
    out = head(x)
    assert out.shape == x.shape


def test_residual_head_unet_accepts_contrast_embedding() -> None:
    """FiLM modulation must accept and consume a contrast embedding."""
    from mriforge.models.generators.latent_diffusion_with_residual_head import (
        _ResidualHeadUNet,
    )

    head = _ResidualHeadUNet(
        in_channels=1, out_channels=1, base_channels=8, depth=2, contrast_dim=4
    )
    x = torch.randn(2, 1, 16, 16)
    emb = torch.randn(2, 4)
    out = head(x, emb)
    assert out.shape == x.shape
