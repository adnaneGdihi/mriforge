"""Tests for ``BaseGANBlock``.

Targets ``spectramr.models.blocks.base``. Abstract base class for GAN blocks
that combines ``nn.Module`` with the project's ``IModel`` interface.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.blocks.base import BaseGANBlock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_base_gan_block_is_nn_module() -> None:
    """``BaseGANBlock`` is a subclass of ``nn.Module``."""
    assert issubclass(BaseGANBlock, nn.Module)


def test_default_forward_raises_not_implemented() -> None:
    """Calling ``forward`` on the base class raises ``NotImplementedError``."""
    block = BaseGANBlock()
    with pytest.raises(NotImplementedError, match="must implement"):
        block(torch.zeros(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# Properties on a subclass
# ---------------------------------------------------------------------------


class _MySubBlock(BaseGANBlock):
    """Concrete subclass for testing inherited helpers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_subclass_name_returns_class_name() -> None:
    """``name`` property returns the subclass name."""
    block = _MySubBlock()
    assert block.name == "_MySubBlock"


def test_subclass_get_parameter_count_sums_trainable_params() -> None:
    """``get_parameter_count`` sums ``numel`` over trainable parameters."""
    block = _MySubBlock()
    expected = (
        4 * 8 + 8     # Linear weight + bias
        + 1 * 4 * 3 * 3 + 4  # Conv2d weight + bias
    )
    assert block.get_parameter_count() == expected


def test_subclass_excludes_frozen_parameters() -> None:
    """Frozen (``requires_grad=False``) parameters are excluded."""
    block = _MySubBlock()
    # Freeze the linear layer.
    for p in block.linear.parameters():
        p.requires_grad_(False)
    # Now only the conv parameters should count.
    expected = 1 * 4 * 3 * 3 + 4
    assert block.get_parameter_count() == expected


def test_subclass_forward_implemented() -> None:
    """A subclass that overrides ``forward`` does not raise."""
    block = _MySubBlock()
    x = torch.randn(1, 1, 4, 4)
    out = block(x)
    assert out.shape == (1, 4, 4, 4)
