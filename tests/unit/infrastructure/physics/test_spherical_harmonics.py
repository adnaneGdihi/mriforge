"""Tests for the 2-D spherical-harmonic (polynomial) basis utilities.

Targets :mod:`spectramr.infrastructure.physics.spherical_harmonics`. The
module fits a B₀ field map to a real polynomial basis up to a given
order — the basis underlies the Koolstra-2021 shim-extrapolation step
used in our ULF pipeline.

Properties exercised here:

* The 2-D polynomial basis matrix has the documented shape
  (``[H*W, K]`` with ``K`` triangular in ``max_order``).
* Fitting a deterministic polynomial recovers the planted coefficients
  within floating-point tolerance.
* The fitted field round-trips: ``evaluate_sh_field(coeffs) ≈ b0``
  when the planted field is itself a polynomial of the same order.
* Mask path: when a non-trivial mask is supplied the in-mask error is
  zero; out-of-mask values come from extrapolation.
* Loud-fail: 3-D inputs raise ``ValueError`` (no silent broadcast).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.spherical_harmonics import (
    _build_2d_polynomial_basis,
    evaluate_sh_field,
    fit_spherical_harmonics,
)


# ── Basis builder ──────────────────────────────────────────────────


class TestPolynomialBasis:
    def test_basis_shape_first_order(self) -> None:
        # Order 1: monomials {1, x, y} → K=3.
        B = _build_2d_polynomial_basis((4, 4), max_order=1)
        assert B.shape == (16, 3)

    def test_basis_shape_second_order(self) -> None:
        # Order 2: {1, x, y, x², xy, y²} → K=6.
        B = _build_2d_polynomial_basis((4, 4), max_order=2)
        assert B.shape == (16, 6)

    def test_basis_shape_third_order(self) -> None:
        # Order 3: K = 1+2+3+4 = 10.
        B = _build_2d_polynomial_basis((4, 4), max_order=3)
        assert B.shape == (16, 10)

    def test_basis_first_column_is_constant_one(self) -> None:
        """The leading basis function is the constant ``1``."""
        B = _build_2d_polynomial_basis((4, 4), max_order=2)
        assert torch.allclose(B[:, 0], torch.ones(16))

    def test_basis_grid_corners_are_pm_one(self) -> None:
        """Coordinates span ``[-1, 1]`` — the linear basis columns hit
        the ±1 boundary in their respective dimension at the corners."""
        B = _build_2d_polynomial_basis((4, 4), max_order=1)
        # Column ordering (per source): order 0 → 1, order 1 → {x_only,
        # y_only} via p=0..order. p=0 means y (xx^0 * yy^1), p=1 means
        # x (xx^1 * yy^0). So col 1 = y, col 2 = x.
        assert B[:, 1].abs().max().item() == pytest.approx(1.0)
        assert B[:, 2].abs().max().item() == pytest.approx(1.0)


# ── Round-trip on a planted polynomial ─────────────────────────────


class TestFitRoundTrip:
    @pytest.fixture
    def grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        # H = W = 16; grid coords in [-1, 1].
        y = torch.linspace(-1, 1, 16)
        x = torch.linspace(-1, 1, 16)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return xx, yy

    def test_constant_field_recovered_exactly(
        self, grid: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        xx, _yy = grid
        b0 = torch.full_like(xx, fill_value=2.5)
        fitted, coeffs = fit_spherical_harmonics(b0, max_order=0)
        # Constant → single coefficient = 2.5.
        assert coeffs.shape == (1,)
        assert coeffs.item() == pytest.approx(2.5, abs=1e-6)
        # Fitted map equals input within numerical tolerance.
        assert torch.allclose(fitted.squeeze(), b0, atol=1e-5)

    def test_linear_field_recovered(
        self, grid: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        xx, yy = grid
        # b0 = 1.0 + 3 * x + 2 * y; order 1 basis order is [1, y, x].
        b0 = 1.0 + 3.0 * xx + 2.0 * yy
        fitted, coeffs = fit_spherical_harmonics(b0, max_order=1)
        assert coeffs.shape == (3,)
        # Compare components with mild numerical tolerance.
        assert coeffs[0].item() == pytest.approx(1.0, abs=1e-5)
        assert coeffs[1].item() == pytest.approx(2.0, abs=1e-5)  # y
        assert coeffs[2].item() == pytest.approx(3.0, abs=1e-5)  # x
        assert torch.allclose(fitted.squeeze(), b0, atol=1e-5)

    def test_quadratic_field_recovered(
        self, grid: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        xx, yy = grid
        # Pure quadratic: x² - y².
        b0 = xx**2 - yy**2
        fitted, _ = fit_spherical_harmonics(b0, max_order=2)
        # Round-trip on the same basis must reconstruct exactly.
        assert torch.allclose(fitted.squeeze(), b0, atol=1e-5)

    def test_evaluate_sh_field_round_trip(
        self, grid: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        xx, yy = grid
        b0 = 0.5 - xx + 2 * yy
        _, coeffs = fit_spherical_harmonics(b0, max_order=1)
        # Re-evaluate; must reproduce the original field.
        reconstructed = evaluate_sh_field(coeffs, (16, 16), max_order=1)
        assert torch.allclose(reconstructed.squeeze(), b0, atol=1e-5)


# ── Mask path ──────────────────────────────────────────────────────


class TestFitWithMask:
    def test_mask_restricted_fit_extrapolates_outside(self) -> None:
        H = W = 16
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        # Planted linear field everywhere.
        b0 = 1.0 + 2.0 * xx
        # Corrupt the *outside* of the mask, then fit only inside the
        # mask — the polynomial fit should still find the true linear
        # coefficients and extrapolate them outside, ignoring the
        # corruption.
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[4:12, 4:12] = True
        corrupted = b0.clone()
        corrupted[~mask] = -999.0

        fitted, _ = fit_spherical_harmonics(
            corrupted, max_order=1, mask=mask.float()
        )
        # Inside the mask the fit matches the true field within tol.
        assert torch.allclose(
            fitted.squeeze()[mask], b0[mask], atol=1e-4
        )
        # Extrapolation outside the mask must NOT keep the corrupted
        # -999 values; it should match the true linear field.
        assert torch.allclose(
            fitted.squeeze()[~mask], b0[~mask], atol=1e-4
        )


# ── Loud-fail on bad input ─────────────────────────────────────────


class TestInputValidation:
    def test_3d_input_raises_value_error(self) -> None:
        # 3-D input must fail loud, not silently broadcast.
        bad = torch.zeros(1, 16, 16)
        with pytest.raises(ValueError, match="Expected 2D or 4D"):
            fit_spherical_harmonics(bad)

    def test_5d_input_raises_value_error(self) -> None:
        bad = torch.zeros(1, 1, 1, 16, 16)
        with pytest.raises(ValueError, match="Expected 2D or 4D"):
            fit_spherical_harmonics(bad)

    def test_4d_input_accepted(self) -> None:
        # The supported [1, 1, H, W] layout must NOT raise.
        b0 = torch.zeros(1, 1, 16, 16)
        fitted, _ = fit_spherical_harmonics(b0, max_order=0)
        assert fitted.shape == (1, 1, 16, 16)


# ── Module exports ─────────────────────────────────────────────────


def test_module_has_documented_exports() -> None:
    import spectramr.infrastructure.physics.spherical_harmonics as mod

    assert "fit_spherical_harmonics" in mod.__all__
    assert "evaluate_sh_field" in mod.__all__
