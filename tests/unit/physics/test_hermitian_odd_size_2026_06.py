"""Regression: Hermitian symmetrisation must be correct for odd-sized grids.

``_cyclic_mirror`` hardcoded ``torch.roll(flip(Z), shifts=(1, 1))`` — the
mirror-index map for an EVEN-length centered (fftshift) spectrum, where the lone
unpaired Nyquist bin sits at index 0. For an ODD length every frequency has a
symmetric partner and the correct mirror of index ``i`` is the plain flip
``(N-1-i)`` with NO roll. The pre-fix code therefore mis-paired every bin on odd
grids, so ``enforce_hermitian`` did not project onto the Hermitian subspace and
``is_hermitian`` mis-classified a genuinely-Hermitian spectrum.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.hermitian import enforce_hermitian, is_hermitian


def _centered_kspace(x: torch.Tensor) -> torch.Tensor:
    """Centered k-space (DC at H//2, W//2) of a real image — the module's
    convention."""
    return torch.fft.fftshift(torch.fft.fft2(x))


def test_real_image_kspace_is_hermitian_even():
    x = torch.randn(8, 8)
    assert is_hermitian(_centered_kspace(x), atol=1e-4)


def test_real_image_kspace_is_hermitian_odd():
    x = torch.randn(9, 9)
    assert is_hermitian(_centered_kspace(x), atol=1e-4)


def test_real_image_kspace_is_hermitian_mixed_parity():
    x = torch.randn(8, 9)
    assert is_hermitian(_centered_kspace(x), atol=1e-4)


def test_enforce_hermitian_yields_real_image_odd():
    k = torch.randn(9, 9) + 1j * torch.randn(9, 9)
    ks = enforce_hermitian(k)
    img = torch.fft.ifft2(torch.fft.ifftshift(ks))
    assert img.imag.abs().max().item() < 1e-5


def test_enforce_hermitian_yields_real_image_even():
    k = torch.randn(8, 8) + 1j * torch.randn(8, 8)
    ks = enforce_hermitian(k)
    img = torch.fft.ifft2(torch.fft.ifftshift(ks))
    assert img.imag.abs().max().item() < 1e-5
