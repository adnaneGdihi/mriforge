"""Tests for ``mriforge.core.types`` protocols and aliases.

Targets ``mriforge.core.types``. Verifies the type aliases (``Image``,
``KSpace``, ``Mask``, ``CoilSens``) are usable as type hints and the
protocols (``IMetrics``, ``IForwardModel``) admit objects implementing
the documented signatures.
"""

from __future__ import annotations

import torch

from mriforge.core.types import (
    CoilSens,
    Device,
    IForwardModel,
    IMetrics,
    Image,
    KSpace,
    Mask,
    Scalar,
    T,
    Tensor,
)


def test_tensor_alias_is_torch_tensor() -> None:
    """``Tensor`` is an alias for ``torch.Tensor``."""
    assert Tensor is torch.Tensor


def test_image_kspace_mask_coilsens_aliases_are_tensor_aliases() -> None:
    """All shape-named aliases resolve to ``torch.Tensor`` (no isinstance branching)."""
    assert Image is torch.Tensor
    assert KSpace is torch.Tensor
    assert Mask is torch.Tensor
    assert CoilSens is torch.Tensor


def test_scalar_alias_admits_int_and_float() -> None:
    """``Scalar`` is a Union of int and float."""
    # Quick runtime check: the union signature should be compatible with both.
    s_int: Scalar = 5
    s_float: Scalar = 5.0
    assert isinstance(s_int, (int, float))
    assert isinstance(s_float, (int, float))


def test_device_alias_admits_str_and_torch_device() -> None:
    """``Device`` admits both ``str`` and ``torch.device``."""
    d_str: Device = "cpu"
    d_obj: Device = torch.device("cpu")
    assert isinstance(d_str, str)
    assert isinstance(d_obj, torch.device)


def test_imetrics_protocol_admits_callable_with_pred_target() -> None:
    """An object with ``__call__(pred, target)`` satisfies ``IMetrics``."""

    class _MyMetric:
        def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (pred - target).abs().mean()

    metric: IMetrics = _MyMetric()
    out = metric(torch.zeros(4), torch.ones(4))
    assert torch.allclose(out, torch.tensor(1.0))


def test_iforwardmodel_protocol_admits_callable_with_forward() -> None:
    """An object with ``forward(x, **kwargs)`` satisfies ``IForwardModel``."""

    class _MyForward:
        def forward(self, x: torch.Tensor, **kwargs):
            return x * 2

    fwd: IForwardModel = _MyForward()
    out = fwd.forward(torch.ones(4))
    assert torch.allclose(out, torch.full((4,), 2.0))


def test_typevar_is_generic() -> None:
    """``T`` is a generic TypeVar (no concrete binding)."""
    from typing import TypeVar
    assert isinstance(T, TypeVar)
