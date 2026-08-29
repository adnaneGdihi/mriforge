"""TRELLIS utility helpers for datatype management.

Migrated from src/implementations/trellis/modules/utils.py
during consolidation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

try:  # pragma: no cover - optional sparse primitives
    from . import sparse as sp  # type: ignore
except ImportError:  # pragma: no cover - sparse ops unavailable
    sp = None


def _iter_parameters(module: nn.Module) -> Iterable[torch.nn.Parameter]:
    """_iter_parameters.

    Args:
        module (nn.Module): Description.
    Returns:
        Iterable[torch.nn.Parameter]: Description.
    """
    yield from module.parameters(recurse=False)
    for child in module.children():
        yield from _iter_parameters(child)


def _convert_module_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """_convert_module_dtype.

    Args:
        module (nn.Module): Description.
        dtype (torch.dtype): Description.
    """
    for param in _iter_parameters(module):
        if param.is_floating_point():
            param.data = param.data.to(dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype)


def _build_fp16_modules() -> tuple[type, ...]:
    """_build_fp16_modules.

    Returns:
        tuple[type, ...]: Description.
    """
    base: tuple[type, ...] = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    )
    if sp is None:
        return base
    sparse_modules = (
        getattr(sp, "SparseConv3d", None),
        getattr(sp, "SparseInverseConv3d", None),
        getattr(sp, "SparseLinear", None),
    )
    return tuple(m for m in (*base, *sparse_modules) if m is not None)


_FP16_MODULES = _build_fp16_modules()


def convert_module_to_f16(module: nn.Module) -> None:
    """Recursively cast supported modules to ``float16``.

    This mirrors the original TRELLIS helper but degrades gracefully when
    sparse kernels are unavailable.
    """

    if isinstance(module, _FP16_MODULES):
        _convert_module_dtype(module, torch.float16)


def convert_module_to_f32(module: nn.Module) -> None:
    """Recursively cast supported modules back to ``float32``."""

    if isinstance(module, _FP16_MODULES):
        _convert_module_dtype(module, torch.float32)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero all parameters in ``module`` in-place and return it."""

    for param in module.parameters():
        param.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale all parameters in ``module`` by ``scale`` and return it."""

    for param in module.parameters():
        param.detach().mul_(scale)
    return module


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply adaptive layer normalisation modulation."""

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


__all__ = [
    "convert_module_to_f16",
    "convert_module_to_f32",
    "modulate",
    "scale_module",
    "zero_module",
]
