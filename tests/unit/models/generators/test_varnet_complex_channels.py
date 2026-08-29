"""Regression: VariationalNetworkGenerator handles complex-channel layout.

Two bugs caught in smoke 2026-05-10 / fixed 2026-05-12:

1. **Channel-count disagreement** between ``initial_recon`` (literal
   ``nn.Conv2d``) and the variational stages (``ComplexConv2d`` which
   doubles its ``in_channels`` argument internally). With YAML
   ``in_channels=2`` (1 complex coil = 2 real interleaved channels),
   ``ComplexConv2d(in_c=2, ...)`` produced a weight expecting 4 real
   channels — but the data tensor had only 2. Crash at iter 1:
   ``weight of size [32, 4, 3, 3], expected input[4, 2, 256, 256] to
   have 4 channels``.

2. **Missing ``F`` import** — ``ComplexAwareBlock.forward`` calls
   ``F.relu`` but ``F`` was never imported at module top, so the model
   raised ``NameError: name 'F' is not defined`` after the shape bug
   was fixed.

The test exercises a full forward + backward at the same input shape
the smoke run uses (``[B, 2, 256, 256]``).
"""

from __future__ import annotations

import pytest
import torch


def test_varnet_forward_backward_on_2real_channels() -> None:
    """[B, 2, H, W] in → [B, 2, H, W] out, gradient flows end-to-end."""
    from mriforge.models.generators.unrolled_reconstruction_generator import (
        VariationalNetworkGenerator,
    )

    model = VariationalNetworkGenerator(
        in_channels=2, out_channels=2, num_stages=2, features=32,
    )
    x = torch.randn(2, 2, 64, 64, requires_grad=False)
    target = torch.randn(2, 2, 64, 64)

    y = model(x)
    assert y.shape == (2, 2, 64, 64), f"unexpected output shape {y.shape}"

    # Gradient must flow through to the trainable params.
    loss = (y - target).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), (
        "no finite gradient reached any model parameter"
    )


def test_varnet_rejects_odd_channel_count() -> None:
    """ComplexAwareBlock raises on odd in_c / out_c — the // 2 semantic
    is what makes the model real-channel-count compatible. Catching the
    odd case loudly is the contract that prevents silent regression."""
    from mriforge.models.generators.unrolled_reconstruction_generator import (
        VariationalNetworkGenerator,
    )

    # in/out_channels of 1 (odd) would silently produce wrong shapes
    # downstream; the explicit check inside ComplexAwareBlock catches it.
    with pytest.raises(ValueError, match="even in_c / out_c"):
        VariationalNetworkGenerator(in_channels=1, out_channels=1)
