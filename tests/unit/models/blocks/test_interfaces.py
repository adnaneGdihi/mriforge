"""Tests for the block-interface ABCs.

Targets ``mriforge.models.blocks.interfaces`` — ``IBlockStrategy`` and
``IFusionStrategy``.

Categories:

- Abstract methods prevent direct instantiation
- A subclass overriding ``forward`` instantiates and runs
- ``IFusionStrategy`` extends ``IBlockStrategy`` (issubclass)
- A fusion subclass passing a list of tensors works
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.interfaces import IBlockStrategy, IFusionStrategy


# ---------------------------------------------------------------------------
# Abstract instantiation
# ---------------------------------------------------------------------------


def test_iblock_strategy_cannot_be_instantiated() -> None:
    """Direct instantiation of ``IBlockStrategy`` raises (abstract)."""
    with pytest.raises(TypeError, match="abstract"):
        IBlockStrategy()  # type: ignore[abstract]


def test_ifusion_strategy_cannot_be_instantiated() -> None:
    """Direct instantiation of ``IFusionStrategy`` raises (abstract)."""
    with pytest.raises(TypeError, match="abstract"):
        IFusionStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Subclass behaviour
# ---------------------------------------------------------------------------


def test_concrete_block_subclass_runs() -> None:
    """A subclass overriding ``forward`` is instantiable and runs."""

    class _Identity(IBlockStrategy):
        def forward(self, x, *args, **kwargs):
            return x

    block = _Identity()
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(block(x), x)


def test_concrete_fusion_subclass_runs() -> None:
    """A fusion subclass takes a list of tensors and returns one tensor."""

    class _Sum(IFusionStrategy):
        def forward(self, inputs, weights=None, **kwargs):
            stacked = torch.stack(inputs, dim=0)
            return stacked.sum(dim=0)

    fusion = _Sum()
    a = torch.randn(1, 4, 8, 8)
    b = torch.randn(1, 4, 8, 8)
    out = fusion([a, b])
    assert torch.allclose(out, a + b)


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_ifusion_strategy_is_subclass_of_iblock_strategy() -> None:
    """``IFusionStrategy`` extends ``IBlockStrategy``."""
    assert issubclass(IFusionStrategy, IBlockStrategy)
