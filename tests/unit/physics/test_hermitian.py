"""Unit tests for src/infrastructure/physics/hermitian.py — Phase 0 of audit_plan_novel."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.hermitian import (
    enforce_hermitian,
    hermitian_gaussian_noise,
    is_hermitian,
)


def _random_real_image(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_enforce_hermitian_is_idempotent() -> None:
    """A second projection must not change the result."""
    K = torch.complex(_random_real_image((4, 1, 16, 16)), _random_real_image((4, 1, 16, 16), seed=1))
    K1 = enforce_hermitian(K)
    K2 = enforce_hermitian(K1)
    assert torch.allclose(K1, K2, atol=1e-6)


def test_enforce_hermitian_passes_hermitian_check() -> None:
    K = torch.complex(_random_real_image((2, 1, 32, 32)), _random_real_image((2, 1, 32, 32), seed=2))
    K_sym = enforce_hermitian(K)
    assert is_hermitian(K_sym, atol=1e-5)


def test_enforce_hermitian_round_trip_real() -> None:
    """ifft(enforce_hermitian(fft(real_image))) must be ~real."""
    img = _random_real_image((1, 1, 16, 16))
    K = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"), dim=(-2, -1))
    K_sym = enforce_hermitian(K)
    img_back = torch.fft.ifft2(torch.fft.ifftshift(K_sym, dim=(-2, -1)), norm="ortho")
    assert img_back.imag.abs().max() < 1e-5


def test_interleaved_layout_supported() -> None:
    """[..., 2] real layout returns same layout."""
    K_interleaved = _random_real_image((2, 1, 8, 8, 2))
    K_sym = enforce_hermitian(K_interleaved)
    assert K_sym.shape == K_interleaved.shape
    assert K_sym.dtype == K_interleaved.dtype


def test_hermitian_noise_is_hermitian() -> None:
    z = hermitian_gaussian_noise((3, 1, 16, 16), dtype=torch.complex64)
    assert is_hermitian(z, atol=1e-5)


def test_rejects_unknown_dtype() -> None:
    with pytest.raises(TypeError):
        enforce_hermitian(torch.zeros(2, 1, 8, 8))  # 4-D real without trailing-2
