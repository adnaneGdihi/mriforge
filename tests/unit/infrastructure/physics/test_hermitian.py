"""Tests for Hermitian-symmetry utilities on centered k-space.

The load-bearing invariant: ``fft2c`` of a real image is Hermitian-symmetric,
and ``enforce_hermitian`` must recognise that (``is_hermitian`` True) and act as
an idempotent projector whose inverse FFT is purely real. A prior bug used the
wrong cyclic-roll on ODD-length axes (``shift=0`` instead of ``shift=2`` for the
``fft2c`` centering), so ``is_hermitian(fft2c(real))`` returned False on odd
grids and ``hermitian_gaussian_noise`` leaked energy into the imaginary channel.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.hermitian import (
    enforce_hermitian,
    hermitian_gaussian_noise,
    is_hermitian,
)


@pytest.mark.physics
class TestCyclicMirrorMatchesFft2c:
    @pytest.mark.parametrize("n", [7, 8, 9, 16])
    def test_fft2c_of_real_image_is_hermitian(self, n: int) -> None:
        # Regression for the odd-grid mirror bug: this was False for odd n.
        torch.manual_seed(0)
        img = torch.randn(1, 1, n, n)
        k = fft2c(img)[0, 0]
        assert is_hermitian(k, atol=1e-4)

    @pytest.mark.parametrize("n", [7, 8, 9, 16])
    def test_enforce_is_noop_on_already_hermitian(self, n: int) -> None:
        torch.manual_seed(1)
        k = fft2c(torch.randn(1, 1, n, n))[0, 0]
        projected = enforce_hermitian(k)
        assert torch.allclose(k, projected, atol=1e-4)

    @pytest.mark.parametrize("n", [7, 8])
    def test_enforced_spectrum_has_real_inverse(self, n: int) -> None:
        # Projecting an arbitrary complex spectrum then inverse-FFT must be real.
        torch.manual_seed(2)
        k = torch.complex(torch.randn(1, 1, n, n), torch.randn(1, 1, n, n))
        sym = enforce_hermitian(k[0, 0]).reshape(1, 1, n, n)
        img = ifft2c(sym)
        assert img.imag.abs().max().item() < 1e-4


@pytest.mark.physics
class TestHermitianGaussianNoise:
    @pytest.mark.parametrize("n", [7, 8, 9])
    def test_sampled_noise_is_hermitian(self, n: int) -> None:
        gen = torch.Generator().manual_seed(3)
        z = hermitian_gaussian_noise((n, n), generator=gen)
        assert is_hermitian(z, atol=1e-4)
        # And therefore has a (numerically) real inverse FFT.
        img = ifft2c(z.reshape(1, 1, n, n))
        assert img.imag.abs().max().item() < 1e-4

    def test_rejects_non_complex_dtype(self) -> None:
        with pytest.raises(ValueError):
            hermitian_gaussian_noise((4, 4), dtype=torch.float32)
