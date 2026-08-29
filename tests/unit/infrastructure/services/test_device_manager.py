"""Tests for ``DeviceManager``.

Targets ``mriforge.infrastructure.services.device_manager``. Wraps
``DevicePolicy`` + a `torch.device` selection for clean DI consumption.

Categories:

- Construction: defaults to CPU when no preference; respects ``cpu``;
  CUDA preference falls back when CUDA is unavailable
- ``get_device`` returns the cached `torch.device`
- ``set_device``: rejects unknown strings; raises if CUDA requested
  without availability
- ``list_devices``: always lists ``"cpu"``
- ``move_to_device``: passes through the policy
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.services.device_manager import DeviceManager


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction_auto_detects() -> None:
    """No explicit preference → auto-detect (CUDA if available, else CPU).

    Documented behaviour: ``preferred_device=None`` is the only ambiguous
    input and means "auto-detect"; the manager picks cuda when CUDA is
    visible to PyTorch and falls back to cpu otherwise. (Explicitly
    requesting a device is the path that raises — see
    ``test_cuda_unavailable_raises``.)
    """
    manager = DeviceManager()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert manager.get_device().type == expected


def test_explicit_cpu_preference() -> None:
    """``preferred_device='cpu'`` → cpu device."""
    manager = DeviceManager(preferred_device="cpu")
    assert manager.get_device().type == "cpu"


def test_cuda_preference_falls_back_when_unavailable() -> None:
    """When CUDA isn't available, the manager silently falls back to CPU."""
    if torch.cuda.is_available():
        pytest.skip("Test only meaningful when CUDA is unavailable")
    manager = DeviceManager(preferred_device="cuda")
    assert manager.get_device().type == "cpu"


# ---------------------------------------------------------------------------
# is_cuda_available + list_devices
# ---------------------------------------------------------------------------


def test_is_cuda_available_matches_torch() -> None:
    """``is_cuda_available`` mirrors ``torch.cuda.is_available``."""
    manager = DeviceManager()
    assert manager.is_cuda_available() is torch.cuda.is_available()


def test_list_devices_always_includes_cpu() -> None:
    """CPU is always in the listing (whether or not CUDA is present)."""
    manager = DeviceManager()
    devices = manager.list_devices()
    assert "cpu" in devices


# ---------------------------------------------------------------------------
# set_device
# ---------------------------------------------------------------------------


def test_set_device_to_cpu_succeeds() -> None:
    """``set_device('cpu')`` is always allowed."""
    manager = DeviceManager()
    manager.set_device("cpu")
    assert manager.get_device().type == "cpu"


def test_set_device_unknown_string_raises() -> None:
    """Unknown device string → ``ValueError``."""
    manager = DeviceManager()
    with pytest.raises(ValueError, match="Unsupported device"):
        manager.set_device("tpu")


def test_set_device_cuda_without_availability_raises() -> None:
    """Requesting CUDA when unavailable → ``RuntimeError``."""
    if torch.cuda.is_available():
        pytest.skip("Test only meaningful without CUDA")
    manager = DeviceManager()
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        manager.set_device("cuda")


# ---------------------------------------------------------------------------
# move_to_device
# ---------------------------------------------------------------------------


def test_move_tensor_to_device() -> None:
    """``move_to_device`` returns the tensor on the manager's device."""
    manager = DeviceManager(preferred_device="cpu")
    tensor = torch.randn(2, 3)
    moved = manager.move_to_device(tensor)
    assert moved.device.type == "cpu"
    assert torch.equal(moved, tensor)
