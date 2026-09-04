"""Unit tests for Quantitative Susceptibility Mapping (QSM) pipeline.

Tests verify physics-level correctness:
- Field map round-trip from known phase
- Laplacian unwrapping removes 2π wraps
- V-SHARP zeroes background field
- TKD recovers known susceptibility from analytical field
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.dipole import apply_3d_dipole_kernel
from spectramr.infrastructure.physics.qsm import (
    compute_field_map,
    ilsqr_dipole_inversion,
    laplacian_phase_unwrap,
    tkd_dipole_inversion,
    vsharp_background_removal,
)


class TestIlsqrDipoleInversion:
    """The CG solve must run on the SYMMETRIC normal operator Dᴴ W² D (not the
    non-symmetric W Dᴴ W D) — otherwise CG is not a valid solver."""

    def test_symmetric_cg_improves_on_tkd_init(self) -> None:
        torch.manual_seed(0)
        D = H = W = 12
        # Smooth susceptibility source (band-limited so TKD is a decent init).
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        chi_gt = torch.exp(-(zz**2 + yy**2 + xx**2) / 0.2)[None, None]
        field = apply_3d_dipole_kernel(chi_gt)
        mask = torch.ones_like(chi_gt)

        chi_tkd = tkd_dipole_inversion(field, mask)
        chi_cg = ilsqr_dipole_inversion(field, mask, lambda_reg=1e-4, max_iter=50)

        err_tkd = (chi_tkd - chi_gt).pow(2).mean().sqrt()
        err_cg = (chi_cg - chi_gt).pow(2).mean().sqrt()
        assert torch.isfinite(chi_cg).all()
        # A valid symmetric-operator CG must not diverge past the TKD init.
        assert float(err_cg) <= float(err_tkd) + 1e-6

    def test_shape_preserved(self) -> None:
        x = torch.randn(1, 1, 8, 8, 8)
        out = ilsqr_dipole_inversion(x, torch.ones_like(x), max_iter=3)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestComputeFieldMap:
    """Field map from dual-echo phase."""

    def test_known_linear_phase(self) -> None:
        """ΔB₀ = Δφ / ΔTE should recover a known constant field."""
        D, H, W = 16, 16, 16
        te1, te2 = 0.010, 0.020  # 10 ms, 20 ms
        known_omega = 100.0  # rad/s

        phase_te1 = torch.full((1, 1, D, H, W), known_omega * te1)
        phase_te2 = torch.full((1, 1, D, H, W), known_omega * te2)

        field_map = compute_field_map(phase_te1, phase_te2, te1, te2)
        assert torch.allclose(
            field_map, torch.full_like(field_map, known_omega), atol=1e-3
        )

    def test_shape_preserved(self) -> None:
        """Output shape must match input."""
        shape = (2, 1, 8, 16, 16)
        p1 = torch.randn(shape)
        p2 = torch.randn(shape)
        result = compute_field_map(p1, p2, 0.01, 0.02)
        assert result.shape == shape

    def test_invalid_te_raises(self) -> None:
        """te2 <= te1 should raise ValueError."""
        p = torch.zeros(1, 1, 4, 4, 4)
        with pytest.raises(ValueError, match="must be > te1"):
            compute_field_map(p, p, 0.02, 0.01)

    def test_output_is_angular_frequency_not_tesla(self) -> None:
        """Output is Δφ/ΔTE (rad/s), with NO gamma division (the docstring
        formula is rad/s, not the Tesla form ΔB0 = Δφ/(γ·ΔTE))."""
        te1, te2 = 0.005, 0.010  # ΔTE = 5 ms
        dphi = 0.3  # rad, within (-π, π) so no wrap
        p1 = torch.zeros(1, 1, 1, 4, 4)
        p2 = torch.full((1, 1, 1, 4, 4), dphi)
        fm = compute_field_map(p1, p2, te1, te2)
        expected = dphi / (te2 - te1)  # 60 rad/s (Tesla form would be ~2e-7)
        assert torch.allclose(fm, torch.full_like(fm, expected), atol=1e-4)


class TestLaplacianPhaseUnwrap:
    """Laplacian-based spatial phase unwrapping."""

    def test_smooth_phase_unchanged(self) -> None:
        """A smooth (unwrapped) phase should pass through approximately unchanged."""
        D, H, W = 16, 16, 16
        # Create smooth phase ramp
        z = torch.linspace(-1, 1, D)
        smooth_phase = z.reshape(1, 1, D, 1, 1).expand(1, 1, D, H, W) * 1.5
        unwrapped = laplacian_phase_unwrap(smooth_phase)
        # Should be close to original (up to a global offset)
        residual = unwrapped - unwrapped.mean()
        original_centered = smooth_phase - smooth_phase.mean()
        corr = (residual * original_centered).sum() / (
            residual.norm() * original_centered.norm() + 1e-8
        )
        assert corr.item() > 0.9, f"Correlation too low: {corr.item()}"

    def test_no_nan_output(self) -> None:
        """Output should not contain NaNs."""
        phase = torch.randn(1, 1, 8, 8, 8) * 2.0
        wrapped = torch.angle(torch.exp(1j * phase))
        result = laplacian_phase_unwrap(wrapped)
        assert not torch.isnan(result).any()

    def test_poisson_solve_finite_real_3d(self) -> None:
        """Regression guard for the F-E Poisson-solve cleanup.

        The k-space solve ``phi = F^-1{ F{lap phi} / L(k) }`` (now via the
        clearly-named ``laplacian_phi_k`` numerator, with the dead ``L_k`` alias
        removed) must still return finite, real, shape-preserving output.
        """
        torch.manual_seed(0)
        phase = torch.randn(1, 1, 8, 8, 8) * 2.0
        wrapped = torch.angle(torch.exp(1j * phase))
        out = laplacian_phase_unwrap(wrapped)
        assert out.shape == wrapped.shape
        assert torch.isfinite(out).all()
        assert not out.is_complex()


class TestVSHARPBackgroundRemoval:
    """V-SHARP background field removal."""

    def test_output_shape_preserved(self) -> None:
        """Output shape must match input."""
        shape = (1, 1, 16, 32, 32)
        field = torch.randn(shape)
        mask = torch.ones(shape)
        # Use small radii for fast test
        result = vsharp_background_removal(field, mask, radii=[3])
        assert result.shape == shape

    def test_outside_mask_is_zero(self) -> None:
        """Field outside the mask should be zero."""
        shape = (1, 1, 16, 32, 32)
        field = torch.randn(shape)
        mask = torch.zeros(shape)
        mask[0, 0, 4:12, 8:24, 8:24] = 1.0
        result = vsharp_background_removal(field, mask, radii=[3])
        outside = result[mask < 0.5]
        assert (outside.abs() < 1e-6).all()


class TestTKDDipoleInversion:
    """TKD dipole inversion."""

    def test_recovers_point_source(self) -> None:
        """TKD should approximately recover a known point susceptibility source."""
        shape = (1, 1, 32, 32, 32)
        chi_gt = torch.zeros(shape)
        chi_gt[0, 0, 16, 16, 16] = 1.0  # Point source

        # Forward: compute field from known chi using dipole
        field = apply_3d_dipole_kernel(chi_gt)

        mask = torch.ones(shape)
        chi_recon = tkd_dipole_inversion(field, mask, threshold=0.1)

        # Peak should be at the same location
        peak_idx = chi_recon[0, 0].argmax()
        expected_idx = chi_gt[0, 0].argmax()
        assert peak_idx == expected_idx, "Peak location mismatch"

    def test_shape_preserved(self) -> None:
        """Output shape must match input."""
        shape = (1, 1, 16, 16, 16)
        field = torch.randn(shape)
        mask = torch.ones(shape)
        result = tkd_dipole_inversion(field, mask)
        assert result.shape == shape
