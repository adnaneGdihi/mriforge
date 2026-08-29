"""Unit tests for :mod:`mriforge.shared.utils.model_utils`.

The module currently exposes a single helper, ``maybe_dropout``, which
returns ``nn.Dropout(p)`` when ``p > 0`` and ``nn.Identity()`` otherwise.

Despite the small surface this helper is invoked from dozens of model
builders, so an inadvertent change to the boundary behaviour (``p ==
0.0`` accidentally falling through to ``Dropout(0)``, or negative ``p``
not raising) would silently change inference / training reproducibility.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mriforge.shared.utils.model_utils import maybe_dropout


class TestMaybeDropout:
    def test_positive_returns_dropout(self) -> None:
        layer = maybe_dropout(0.5)
        assert isinstance(layer, nn.Dropout)
        assert layer.p == pytest.approx(0.5)

    def test_zero_returns_identity(self) -> None:
        layer = maybe_dropout(0.0)
        # Strict-inequality boundary — exactly 0 is Identity, not Dropout(0).
        assert isinstance(layer, nn.Identity)

    def test_small_positive_returns_dropout(self) -> None:
        # Any tiny positive value selects Dropout.
        layer = maybe_dropout(1e-6)
        assert isinstance(layer, nn.Dropout)

    def test_negative_returns_identity(self) -> None:
        # ``dropout > 0`` is False for negative values → Identity.
        # (Negative dropout is meaningless; PyTorch would crash inside
        # nn.Dropout — the guard here protects callers.)
        layer = maybe_dropout(-0.1)
        assert isinstance(layer, nn.Identity)

    def test_identity_layer_is_pass_through(self) -> None:
        layer = maybe_dropout(0.0)
        x = torch.randn(2, 3, 4, 4)
        out = layer(x)
        assert torch.equal(out, x)

    def test_dropout_layer_in_eval_mode_is_pass_through(self) -> None:
        # Dropout is a no-op in eval mode regardless of p > 0.
        layer = maybe_dropout(0.9)
        layer.eval()
        x = torch.randn(2, 3, 4, 4)
        out = layer(x)
        assert torch.equal(out, x)
