"""Tests for ``spectramr.shared.utils.device_utils``.

Targets ``ensure_device_consistency``, ``safe_to_device``,
``validate_device_consistency``, ``create_device_aligned_tensor``,
``get_device``, ``get_device_dtype_info``. CPU-only — no CUDA assertions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.shared.utils.device_utils import (
    create_device_aligned_tensor,
    ensure_device_consistency,
    get_device,
    get_device_dtype_info,
    safe_to_device,
    validate_device_consistency,
)


def _tiny_model() -> nn.Module:
    """Tiny linear model with one buffer for device-routing tests."""
    m = nn.Linear(4, 2)
    m.register_buffer("scratch", torch.zeros(2))
    return m


# ---------------------------------------------------------------------------
# ensure_device_consistency
# ---------------------------------------------------------------------------


def test_ensure_device_consistency_moves_params_and_buffers_to_cpu() -> None:
    """Every parameter and buffer ends on the requested device."""
    model = _tiny_model()
    out = ensure_device_consistency(model, "cpu")
    for p in out.parameters():
        assert p.device.type == "cpu"
    for b in out.buffers():
        assert b.device.type == "cpu"


def test_ensure_device_consistency_accepts_torch_device_object() -> None:
    """``torch.device`` instance works the same as the string form."""
    model = _tiny_model()
    out = ensure_device_consistency(model, torch.device("cpu"))
    assert out is model  # mutated in place + returned


# ---------------------------------------------------------------------------
# safe_to_device
# ---------------------------------------------------------------------------


def test_safe_to_device_handles_none() -> None:
    """``None`` input is returned untouched (no AttributeError)."""
    assert safe_to_device(None, "cpu") is None


def test_safe_to_device_moves_tensor() -> None:
    """A bare tensor is moved to the requested device."""
    t = torch.zeros(4)
    out = safe_to_device(t, "cpu")
    assert out.device.type == "cpu"


def test_safe_to_device_with_dtype() -> None:
    """When ``dtype`` is given, output has both the device AND dtype."""
    t = torch.zeros(4, dtype=torch.float32)
    out = safe_to_device(t, "cpu", dtype=torch.float64)
    assert out.dtype == torch.float64
    assert out.device.type == "cpu"


def test_safe_to_device_routes_models_through_ensure() -> None:
    """Modules go through ``ensure_device_consistency`` (params + buffers moved)."""
    model = _tiny_model()
    out = safe_to_device(model, "cpu")
    assert validate_device_consistency(out, "cpu")


# ---------------------------------------------------------------------------
# validate_device_consistency
# ---------------------------------------------------------------------------


def test_validate_device_consistency_returns_true_for_aligned_model() -> None:
    """A freshly-built model on CPU passes validation."""
    model = _tiny_model()
    assert validate_device_consistency(model, "cpu") is True


def test_validate_device_consistency_with_torch_device_index() -> None:
    """Index check passes when the expected index is None."""
    model = _tiny_model()
    assert validate_device_consistency(model, torch.device("cpu")) is True


# ---------------------------------------------------------------------------
# create_device_aligned_tensor
# ---------------------------------------------------------------------------


def test_create_device_aligned_tensor_default_zeros() -> None:
    """Without ``fill_value`` → zeros tensor on the requested device."""
    t = create_device_aligned_tensor((2, 3), "cpu")
    assert t.shape == (2, 3)
    assert t.device.type == "cpu"
    assert torch.all(t == 0)


def test_create_device_aligned_tensor_fill_value() -> None:
    """``fill_value`` propagates to all elements."""
    t = create_device_aligned_tensor((2, 3), "cpu", fill_value=7.5)
    assert torch.all(t == 7.5)


def test_create_device_aligned_tensor_dtype() -> None:
    """Dtype argument is respected."""
    t = create_device_aligned_tensor((2,), "cpu", dtype=torch.int32)
    assert t.dtype == torch.int32


# ---------------------------------------------------------------------------
# get_device / get_device_dtype_info
# ---------------------------------------------------------------------------


def test_get_device_returns_torch_device() -> None:
    """``get_device`` returns a ``torch.device`` instance."""
    d = get_device()
    assert isinstance(d, torch.device)
    assert d.type in {"cpu", "cuda"}


def test_get_device_dtype_info_summarises_model() -> None:
    """Output dict contains ``devices``, ``dtypes``, ``primary_device``, ``primary_dtype``."""
    info = get_device_dtype_info(_tiny_model())
    assert {"devices", "dtypes", "primary_device", "primary_dtype"} <= info.keys()
    assert info["primary_device"].startswith("cpu")
    assert "float32" in info["primary_dtype"] or "float" in info["primary_dtype"]
