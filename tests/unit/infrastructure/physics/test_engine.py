"""Tests for ``PhysicsEngine``.

Targets ``spectramr.infrastructure.physics.engine`` — the unified physics
facade.

Categories:

- Construction: defaults / explicit kwargs persist
- ``device`` property reads the configured device
- ``to(device)`` returns self for chaining + updates the device
- FFT round-trip via ``image_to_kspace → kspace_to_image`` recovers
  the input within tolerance
- ``apply_mask`` zeroes unsampled k-space samples
- SENSE forward+adjoint composes consistently for trivial smaps
- ``get_dc_module`` is lazy and idempotent
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.engine import PhysicsEngine


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default device is CPU; default ``dc_type='soft'``."""
    engine = PhysicsEngine()
    assert engine.device == torch.device("cpu")
    assert engine._config["dc_type"] == "soft"
    assert engine._config["dc_weight"] == 1.0


def test_explicit_construction() -> None:
    """Explicit kwargs round-trip onto the engine."""
    engine = PhysicsEngine(device="cpu", dc_type="hard", dc_weight=0.7)
    assert engine._config["dc_type"] == "hard"
    assert engine._config["dc_weight"] == 0.7


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def test_to_returns_self() -> None:
    """``to(device)`` is fluent."""
    engine = PhysicsEngine()
    out = engine.to("cpu")
    assert out is engine


def test_to_updates_device() -> None:
    """``to`` updates the cached device."""
    engine = PhysicsEngine(device="cpu")
    engine.to(torch.device("cpu"))
    assert engine.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# FFT round-trip
# ---------------------------------------------------------------------------


def test_image_kspace_round_trip() -> None:
    """``ifft2c(fft2c(x)) ≈ x``."""
    engine = PhysicsEngine()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    k = engine.image_to_kspace(img)
    back = engine.kspace_to_image(k)
    assert torch.allclose(back, img, atol=1e-5)


def test_image_kspace_shape_preserved() -> None:
    """FFT operations preserve shape."""
    engine = PhysicsEngine()
    img = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
    k = engine.image_to_kspace(img)
    assert k.shape == img.shape


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------


def test_apply_mask_zeroes_unsampled() -> None:
    """Where mask = 0, k-space samples are zeroed."""
    engine = PhysicsEngine()
    k = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mask = torch.zeros(1, 1, 8, 8)
    masked = engine.apply_mask(k, mask)
    assert masked.abs().max().item() == 0.0


def test_apply_mask_preserves_sampled() -> None:
    """Where mask = 1, k-space samples are preserved."""
    engine = PhysicsEngine()
    k = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    mask = torch.ones(1, 1, 8, 8)
    masked = engine.apply_mask(k, mask)
    assert torch.allclose(masked, k)


# ---------------------------------------------------------------------------
# SENSE forward + adjoint
# ---------------------------------------------------------------------------


def test_sense_forward_returns_complex_kspace() -> None:
    """``sense_forward`` returns a complex multi-coil k-space tensor."""
    engine = PhysicsEngine()
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = engine.sense_forward(img)
    # Single-coil is the trivial case — shape unchanged.
    assert out.shape == img.shape
    assert torch.is_complex(out)


def test_sense_adjoint_returns_complex_image() -> None:
    """``sense_adjoint`` returns a complex image."""
    engine = PhysicsEngine()
    k = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = engine.sense_adjoint(k)
    assert out.shape == k.shape
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# Data-consistency module
# ---------------------------------------------------------------------------


def test_dc_module_is_lazy() -> None:
    """``_dc_module`` starts as None until first call."""
    engine = PhysicsEngine()
    assert engine._dc_module is None


def test_get_dc_module_idempotent() -> None:
    """``get_dc_module`` returns the same module on repeated calls."""
    engine = PhysicsEngine()
    a = engine.get_dc_module()
    b = engine.get_dc_module()
    assert a is b
