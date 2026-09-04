"""Unit tests for models/stability/runtime_error_handler.py.

Covers: RuntimeErrorHandler static helpers (safe_lower, ensure_device,
        safe_slice_key, handle_gradient_error).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.stability.runtime_error_handler import RuntimeErrorHandler


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_safe_lower_canary():
    """safe_lower returns lowercase string for a plain string."""
    assert RuntimeErrorHandler.safe_lower("Hello World") == "hello world"


# ---------------------------------------------------------------------------
# Parametrized: safe_lower
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "obj,expected",
    [
        ("ABC", "abc"),
        (None, ""),
        (123, "123"),
        ("already_lower", "already_lower"),
        ("MixedCase", "mixedcase"),
    ],
)
def test_safe_lower_variants(obj, expected):
    assert RuntimeErrorHandler.safe_lower(obj) == expected


# ---------------------------------------------------------------------------
# Parametrized: safe_slice_key
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "slice_obj,expected_prefix",
    [
        (slice(0, 10, 2), "slice_"),
        ("some_string", "some_string"),
        (42, "42"),
    ],
)
def test_safe_slice_key(slice_obj, expected_prefix):
    key = RuntimeErrorHandler.safe_slice_key(slice_obj)
    assert str(key).startswith(expected_prefix)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ensure_device_none_returns_none():
    """ensure_device passes None through unchanged."""
    result = RuntimeErrorHandler.ensure_device(None, torch.device("cpu"))
    assert result is None


@pytest.mark.unit
def test_ensure_device_non_tensor_returned_unchanged():
    """Non-tensor objects without .device attribute pass through unchanged."""
    obj = {"key": "value"}
    result = RuntimeErrorHandler.ensure_device(obj, torch.device("cpu"))
    assert result is obj


@pytest.mark.unit
def test_ensure_device_same_device_no_copy():
    """Tensor already on target device is returned as-is."""
    t = torch.zeros(3)  # cpu
    result = RuntimeErrorHandler.ensure_device(t, torch.device("cpu"))
    assert result.data_ptr() == t.data_ptr()


@pytest.mark.unit
def test_handle_gradient_error_calls_zero_grad():
    """handle_gradient_error clears gradients on a model that supports zero_grad."""
    model = nn.Linear(3, 3)
    # Give the model some gradients
    out = model(torch.randn(2, 3))
    out.sum().backward()
    # Should run without error and clear gradients
    RuntimeErrorHandler.handle_gradient_error(model, RuntimeError("test"))
    for p in model.parameters():
        assert p.grad is None


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_safe_model_call_reraises_non_device_error():
    """safe_model_call must re-raise RuntimeError unrelated to device."""
    def bad_model(*args, **kwargs):
        raise RuntimeError("totally unrelated error")

    with pytest.raises(RuntimeError, match="totally unrelated error"):
        RuntimeErrorHandler.safe_model_call(bad_model)
