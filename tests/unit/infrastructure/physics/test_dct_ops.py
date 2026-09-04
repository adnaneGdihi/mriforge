"""Tests for the 2D Neumann DCT operations.

Covers the DCT-IDCT round trip (the load-bearing invariant for the
blurring-diffusion forward process), orthonormality / energy
preservation, shape/dtype/device preservation, and AMP safety.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.dct_ops import (
    dct1d,
    dct2_neumann,
    idct1d,
    idct2_neumann,
)


@pytest.mark.physics
class TestDct1d:
    def test_round_trip(self) -> None:
        x = torch.randn(4, 16)
        recon = idct1d(dct1d(x))
        assert torch.allclose(x, recon, atol=1e-4)

    def test_round_trip_odd_length(self) -> None:
        x = torch.randn(3, 17)
        recon = idct1d(dct1d(x))
        assert torch.allclose(x, recon, atol=1e-4)

    def test_energy_preserved_orthonormal(self) -> None:
        x = torch.randn(2, 32)
        coeffs = dct1d(x)
        # Parseval for an orthonormal transform.
        assert torch.allclose(
            (x**2).sum(-1), (coeffs**2).sum(-1), atol=1e-3
        )

    def test_idct1d_matches_scipy_ortho_inverse(self) -> None:
        # Regression: idct1d previously dropped the Makhoul half-spectrum
        # imaginary term, so it was NOT the inverse of dct1d (round-trip was a
        # ~0.45x non-constant distortion). Pin both forward and inverse against
        # scipy's orthonormal DCT-II / IDCT-II.
        sfft = pytest.importorskip("scipy.fft")
        for n in (7, 8, 16, 17):
            x = torch.randn(4, n, dtype=torch.float64)
            ref_fwd = sfft.dct(x.numpy(), type=2, norm="ortho", axis=-1)
            assert torch.allclose(
                dct1d(x), torch.from_numpy(ref_fwd), atol=1e-6
            )
            ref_inv = sfft.idct(ref_fwd, type=2, norm="ortho", axis=-1)
            assert torch.allclose(
                idct1d(torch.from_numpy(ref_fwd)),
                torch.from_numpy(ref_inv),
                atol=1e-6,
            )


@pytest.mark.physics
class TestDct2Neumann:
    def test_round_trip(self) -> None:
        x = torch.randn(2, 3, 32, 32)
        recon = idct2_neumann(dct2_neumann(x))
        assert torch.allclose(x, recon, atol=1e-4)

    def test_round_trip_rectangular(self) -> None:
        x = torch.randn(1, 1, 24, 40)
        recon = idct2_neumann(dct2_neumann(x))
        assert torch.allclose(x, recon, atol=1e-4)

    def test_shape_dtype_preserved(self) -> None:
        x = torch.randn(2, 1, 16, 16, dtype=torch.float64)
        out = dct2_neumann(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_dc_coefficient_is_mean_scaled(self) -> None:
        # A constant image has all energy in the DC coefficient.
        x = torch.ones(1, 1, 8, 8)
        coeffs = dct2_neumann(x)
        # Off-DC coefficients should be ~0.
        off_dc = coeffs.clone()
        off_dc[..., 0, 0] = 0.0
        assert torch.allclose(off_dc, torch.zeros_like(off_dc), atol=1e-4)
        assert coeffs[..., 0, 0].abs().item() > 1.0

    def test_energy_preserved_2d(self) -> None:
        x = torch.randn(1, 1, 16, 16)
        coeffs = dct2_neumann(x)
        assert torch.allclose((x**2).sum(), (coeffs**2).sum(), atol=1e-2)

    def test_amp_safe_no_nan(self) -> None:
        x = torch.randn(2, 1, 16, 16)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            recon = idct2_neumann(dct2_neumann(x))
        assert torch.isfinite(recon).all()
