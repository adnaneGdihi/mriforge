"""Tests for ``GradientCheckpointing.apply_checkpointing``.

Regression coverage for the 2026-06-24 Hyper-Mamba crash: applying generic
gradient checkpointing collapsed every patched module's ``forward`` onto the
*last* checkpointed module because the wrapper lambda captured the loop
variables ``module`` / ``checkpointed_forward`` by reference (Python closure
late-binding). A network whose first conv was ``Conv2d(1, C)`` then crashed
with ``expected input to have C channels, but got 1`` the moment the stem ran
the head's forward. See ``hilbert_mamba`` arms exp_hm_01/03/04/05 (all set
``optimization.gradient.enable_checkpointing: true``).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from spectramr.models.profiling.advanced_profiling import (  # noqa: E402
    GradientCheckpointing,
)


class _AsymmetricConvNet(nn.Module):
    """Two convs with *different* channel signatures.

    The asymmetry is load-bearing: if checkpointing mis-binds the stem's
    forward to the head's (``Conv2d(8, 1, 1)``), the stem receives a
    1-channel input and the channel-count assertion fires — exactly the
    production crash.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.stem(x)))


def test_apply_checkpointing_preserves_per_module_forward() -> None:
    """Each patched module must run ITS OWN forward, not the last one's."""
    model = _AsymmetricConvNet().eval()
    x = torch.randn(2, 1, 16, 16)

    reference = model(x)  # before patching — ground truth

    GradientCheckpointing.apply_checkpointing(model, checkpoint_ratio=1.0)

    out = model(x)  # late-binding bug => crashes here with a channel mismatch
    assert out.shape == reference.shape
    torch.testing.assert_close(out, reference)


def test_apply_checkpointing_propagates_gradients() -> None:
    """Checkpointed forward must still produce gradients for every layer.

    With a non-grad input and reentrant checkpointing, gradients silently
    vanish; the wrapper must use ``use_reentrant=False`` so the stem weights
    receive a gradient.
    """
    model = _AsymmetricConvNet().train()
    GradientCheckpointing.apply_checkpointing(model, checkpoint_ratio=1.0)

    x = torch.randn(2, 1, 16, 16)  # input does NOT require grad (dataloader case)
    model(x).pow(2).mean().backward()

    assert model.stem.weight.grad is not None
    assert torch.isfinite(model.stem.weight.grad).all()
