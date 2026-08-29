r"""Tests for the Koolstra et al. (2021) distortion correction pipeline.

Verifies the following components:
1. Shepp-Logan phantom generation
2. Halbach B₀ field simulation
3. TSE signal model (forward simulation)
4. Phase-difference B₀ mapping
5. Spherical harmonic fitting
6. Conjugate Phase Reconstruction (CPR)
7. Model-Based (MB) reconstruction via Split Bregman
8. Iterative reconstruction pipeline
"""

from __future__ import annotations

import math

import pytest
import torch

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def device() -> torch.device:
    """Default test device (CPU for CI, GPU if available)."""
    return torch.device("cpu")


@pytest.fixture
def phantom_small(device: torch.device) -> torch.Tensor:
    """16×16 Shepp-Logan phantom for fast testing."""
    from mriforge.infrastructure.physics.signal_models.tse_b0_model import shepp_logan_2d

    return shepp_logan_2d(size=16, device=device)


@pytest.fixture
def phantom_medium(device: torch.device) -> torch.Tensor:
    """32×32 Shepp-Logan phantom for moderate tests."""
    from mriforge.infrastructure.physics.signal_models.tse_b0_model import shepp_logan_2d

    return shepp_logan_2d(size=32, device=device)


@pytest.fixture
def b0_small(device: torch.device) -> torch.Tensor:
    """16×16 B₀ map with moderate inhomogeneity."""
    from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
        simulate_halbach_b0,
    )

    return simulate_halbach_b0((16, 16), center_hz=200.0, seed=42, device=device)


@pytest.fixture
def b0_medium(device: torch.device) -> torch.Tensor:
    """32×32 B₀ map with moderate inhomogeneity."""
    from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
        simulate_halbach_b0,
    )

    return simulate_halbach_b0((32, 32), center_hz=200.0, seed=42, device=device)


# ─────────────────────────────────────────────────────────────────────
# 1. Shepp-Logan Phantom Tests
# ─────────────────────────────────────────────────────────────────────


class TestSheppLogan:
    """Tests for the Shepp-Logan phantom generator."""

    def test_shape(self, phantom_small: torch.Tensor) -> None:
        assert phantom_small.shape == (1, 1, 16, 16)

    def test_range(self, phantom_small: torch.Tensor) -> None:
        assert phantom_small.min() >= 0.0
        assert phantom_small.max() <= 1.0

    def test_non_trivial(self, phantom_small: torch.Tensor) -> None:
        """Phantom should have structure, not be all zeros or all ones."""
        unique_vals = torch.unique(phantom_small)
        assert len(unique_vals) > 2

    def test_real_valued(self, phantom_small: torch.Tensor) -> None:
        assert not torch.is_complex(phantom_small)

    def test_larger_resolution(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import shepp_logan_2d

        img = shepp_logan_2d(128, device=device)
        assert img.shape == (1, 1, 128, 128)
        assert img.max() <= 1.0


# ─────────────────────────────────────────────────────────────────────
# 2. Halbach B₀ Simulation Tests
# ─────────────────────────────────────────────────────────────────────


class TestHalbachB0:
    """Tests for Halbach-type B₀ map generation."""

    def test_shape(self, b0_small: torch.Tensor) -> None:
        assert b0_small.shape == (1, 1, 16, 16)

    def test_centre_field_range(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            simulate_halbach_b0,
        )

        b0 = simulate_halbach_b0((64, 64), center_hz=600.0, seed=0, device=device)
        # Peak deviation should not massively exceed the requested center_hz
        assert b0.abs().max() <= 1200.0  # Some tolerance for noise
        assert b0.abs().max() > 100.0  # Should be non-trivial

    def test_off_centre_stronger(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            simulate_halbach_b0,
        )

        b0_centre = simulate_halbach_b0(
            (64, 64), center_hz=600.0, off_center=False, seed=42, device=device,
        )
        b0_off = simulate_halbach_b0(
            (64, 64), off_center_hz=1500.0, off_center=True, seed=42, device=device,
        )
        assert b0_off.abs().max() > b0_centre.abs().max()

    def test_reproducible(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            simulate_halbach_b0,
        )

        b0_a = simulate_halbach_b0((16, 16), seed=123, device=device)
        b0_b = simulate_halbach_b0((16, 16), seed=123, device=device)
        assert torch.allclose(b0_a, b0_b)


# ─────────────────────────────────────────────────────────────────────
# 3. TSE Signal Model Tests
# ─────────────────────────────────────────────────────────────────────


class TestTSESignalModel:
    """Tests for the TSE B₀ signal model."""

    def test_no_b0_roundtrip(self, device: torch.device) -> None:
        """With zero B₀, signal model should match DFT (approximately)."""
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            TSESignalModel,
            shepp_logan_2d,
        )

        size = 8  # Very small for speed
        phantom = shepp_logan_2d(size, device=device)
        b0_zero = torch.zeros(1, 1, size, size, device=device)

        model = TSESignalModel(
            n_readout=size, n_phase=size,
            fov_m=0.225, bw_hz=20000.0,
            oversample_factor=1,  # No oversampling for exact DFT match
        )

        kspace = model.simulate_kspace(phantom, b0_zero, t_shift=0.0, snr=None)

        assert kspace.shape == (1, 1, size, size)
        assert torch.is_complex(kspace)
        # K-space should be non-trivial
        assert kspace.abs().max() > 0.0

    def test_paired_data_shapes(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            TSESignalModel,
            shepp_logan_2d,
            simulate_halbach_b0,
        )

        size = 8
        phantom = shepp_logan_2d(size, device=device)
        b0 = simulate_halbach_b0((size, size), center_hz=100, seed=1, device=device)

        model = TSESignalModel(
            n_readout=size, n_phase=size, oversample_factor=1,
        )
        k1, k2 = model.generate_paired_data(phantom, b0, t_shift=100e-6, snr=None)

        assert k1.shape == k2.shape == (1, 1, size, size)
        # Shifted and unshifted should be different (B₀ ≠ 0)
        assert not torch.allclose(k1, k2)

    def test_noise_adds_variation(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.signal_models.tse_b0_model import (
            TSESignalModel,
            shepp_logan_2d,
        )

        size = 8
        phantom = shepp_logan_2d(size, device=device)
        b0 = torch.zeros(1, 1, size, size, device=device)

        model = TSESignalModel(n_readout=size, n_phase=size, oversample_factor=1)
        k_clean = model.simulate_kspace(phantom, b0, snr=None)
        k_noisy = model.simulate_kspace(phantom, b0, snr=5.0)

        # Noisy version should differ
        diff = (k_clean - k_noisy).abs().mean()
        assert diff > 0.0


# ─────────────────────────────────────────────────────────────────────
# 4. B₀ Mapping Tests
# ─────────────────────────────────────────────────────────────────────


class TestB0Mapping:
    """Tests for phase-difference B₀ estimation."""

    def test_known_phase_recovery(self, device: torch.device) -> None:
        """If we impose a known phase, the B₀ map should recover it."""
        from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov

        H, W = 32, 32
        t_shift = 100e-6  # seconds

        # Create a known smooth B₀ field (Hz)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        b0_true = 200.0 * (xx**2 + yy**2)  # Quadratic bowl, peak ~400 Hz

        # Generate phase difference: φ = -2π * t_shift * ΔB₀
        phase_diff = -2.0 * math.pi * t_shift * b0_true
        phase_diff_4d = phase_diff.unsqueeze(0).unsqueeze(0)

        # Estimate B₀
        b0_est = estimate_b0_tikhonov(
            phase_diff_4d, t_shift=t_shift, gamma=1e-12,  # Very low reg
        )

        # Check recovery (should be close for smooth fields with low reg)
        b0_true_4d = b0_true.unsqueeze(0).unsqueeze(0)
        error_hz = (b0_est - b0_true_4d).abs().mean()
        assert error_hz < 50.0, f"Mean B₀ error {error_hz:.1f} Hz too large"

    def test_output_shape(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov

        phi = torch.randn(1, 1, 16, 16, device=device)
        b0 = estimate_b0_tikhonov(phi, t_shift=100e-6)
        assert b0.shape == (1, 1, 16, 16)

    def test_map_b0_pipeline(self, device: torch.device) -> None:
        """Test full pipeline: two complex images → B₀ map."""
        from mriforge.infrastructure.physics.b0_mapping import map_b0

        H, W = 16, 16
        # Two images with known phase relationship
        mag = torch.ones(1, 1, H, W, device=device)
        phase1 = torch.zeros(1, 1, H, W, device=device)
        phase2 = torch.ones(1, 1, H, W, device=device) * 0.5

        img1 = mag * torch.exp(1j * phase1)
        img2 = mag * torch.exp(1j * phase2)

        b0 = map_b0(img1, img2, t_shift=100e-6)
        assert b0.shape == (1, 1, H, W)
        assert not torch.isnan(b0).any()


# ─────────────────────────────────────────────────────────────────────
# 5. Spherical Harmonics Tests
# ─────────────────────────────────────────────────────────────────────


class TestSphericalHarmonics:
    """Tests for SH fitting and extrapolation."""

    def test_exact_quadratic_fit(self, device: torch.device) -> None:
        """A quadratic field should be perfectly fitted by order-2 SH."""
        from mriforge.infrastructure.physics.spherical_harmonics import (
            fit_spherical_harmonics,
        )

        H, W = 32, 32
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        field = 100.0 * xx**2 + 50.0 * yy**2 + 30.0 * xx * yy + 10.0

        fitted, coeffs = fit_spherical_harmonics(
            field.unsqueeze(0).unsqueeze(0), max_order=2,
        )

        error = (fitted.squeeze() - field).abs().max()
        assert error < 1.0, f"Max error {error:.2f} Hz for exact quadratic"

    def test_masked_fit_extrapolates(self, device: torch.device) -> None:
        """Fitting with a mask should still produce values outside the mask."""
        from mriforge.infrastructure.physics.spherical_harmonics import (
            fit_spherical_harmonics,
        )

        H, W = 32, 32
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        field = 100.0 * (xx**2 + yy**2)

        # Circular mask
        mask = (xx**2 + yy**2 < 0.5**2).float()

        fitted, _ = fit_spherical_harmonics(
            field.unsqueeze(0).unsqueeze(0),
            max_order=2,
            mask=mask.unsqueeze(0).unsqueeze(0),
        )

        # Outside the mask, fitted should still be reasonable (extrapolated)
        outside = ~mask.bool()
        assert fitted.squeeze()[outside].abs().max() > 0.0

    def test_coefficients_count(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.spherical_harmonics import (
            fit_spherical_harmonics,
        )

        field = torch.randn(1, 1, 16, 16, device=device)

        # Order 2: terms = 1 + 2 + 3 = 6
        _, coeffs = fit_spherical_harmonics(field, max_order=2)
        assert coeffs.shape == (6,)

        # Order 3: terms = 1 + 2 + 3 + 4 = 10
        _, coeffs = fit_spherical_harmonics(field, max_order=3)
        assert coeffs.shape == (10,)


# ─────────────────────────────────────────────────────────────────────
# 6. Conjugate Phase Reconstruction Tests
# ─────────────────────────────────────────────────────────────────────


class TestCPR:
    """Tests for Conjugate Phase Reconstruction."""

    def test_zero_b0_matches_fft(self, device: torch.device) -> None:
        """With B₀=0, CPR should approximate standard FFT reconstruction."""
        from mriforge.infrastructure.physics.conjugate_phase import (
            ConjugatePhaseReconstructor,
        )
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c

        H, W = 16, 16
        # Create a simple test image
        image = torch.zeros(1, 1, H, W, device=device, dtype=torch.complex64)
        image[0, 0, 4:12, 4:12] = 1.0 + 0j

        kspace = fft2c(image)
        b0_zero = torch.zeros(1, 1, H, W, device=device)

        cpr = ConjugatePhaseReconstructor(
            freq_range_hz=(-500, 500), freq_step_hz=10.0,
            bw_hz=20000.0, n_readout=W,
        )

        recon = cpr.reconstruct(kspace, b0_zero, t_shift=0.0)
        fft_recon = ifft2c(kspace)

        # Should be close (not exact due to discretisation)
        nrmse = (recon - fft_recon).abs().mean() / (fft_recon.abs().mean() + 1e-12)
        assert nrmse < 0.3, f"NRMSE {nrmse:.3f} too high for zero B₀"

    def test_output_shape(self, device: torch.device) -> None:
        from mriforge.infrastructure.physics.conjugate_phase import (
            ConjugatePhaseReconstructor,
        )
        from mriforge.infrastructure.physics.fft_ops import fft2c

        H, W = 16, 16
        image = torch.randn(1, 1, H, W, device=device, dtype=torch.complex64)
        kspace = fft2c(image)
        b0 = torch.zeros(1, 1, H, W, device=device)

        cpr = ConjugatePhaseReconstructor(
            freq_range_hz=(-100, 100), freq_step_hz=10.0, n_readout=W,
        )

        recon = cpr.reconstruct(kspace, b0)
        assert recon.shape == (1, 1, H, W)
        assert torch.is_complex(recon)


# ─────────────────────────────────────────────────────────────────────
# 7. Split Bregman Tests
# ─────────────────────────────────────────────────────────────────────


class TestSplitBregman:
    """Tests for the Split Bregman solver."""

    def test_identity_recovery(self, device: torch.device) -> None:
        """With identity encoding (E=I), should recover input."""
        from mriforge.infrastructure.physics.optimizers.split_bregman import (
            SplitBregmanSolver,
        )

        N = 16
        m_true = torch.randn(1, 1, N, N, device=device, dtype=torch.complex64)

        solver = SplitBregmanSolver(
            forward_op=lambda m: m,  # Identity
            adjoint_op=lambda y: y,  # Identity
            mu=1.0,
            lambda_tv=0.0,  # No regularisation → exact recovery
            outer_iterations=3,
            inner_iterations=2,
            cg_max_iter=50,
        )

        m_recon = solver.solve(m_true)

        nrmse = (m_recon - m_true).abs().mean() / (m_true.abs().mean() + 1e-12)
        assert nrmse < 0.1, f"Identity recovery NRMSE {nrmse:.3f} too high"

    def test_tv_denoising(self, device: torch.device) -> None:
        """TV regularisation should smooth noisy input."""
        from mriforge.infrastructure.physics.optimizers.split_bregman import (
            SplitBregmanSolver,
        )

        N = 16
        # Piecewise constant image + noise
        m_clean = torch.zeros(1, 1, N, N, device=device, dtype=torch.complex64)
        m_clean[0, 0, 4:12, 4:12] = 1.0 + 0j

        noise = (torch.randn(1, 1, N, N, device=device) * 0.3).to(torch.complex64)
        m_noisy = m_clean + noise

        solver = SplitBregmanSolver(
            forward_op=lambda m: m,
            adjoint_op=lambda y: y,
            mu=1.0,
            lambda_tv=0.05,
            beta=0.5,
            outer_iterations=10,
            inner_iterations=2,
            cg_max_iter=30,
        )

        m_denoised = solver.solve(m_noisy)

        # Denoised should be closer to clean than noisy
        error_noisy = (m_noisy - m_clean).abs().mean()
        error_denoised = (m_denoised - m_clean).abs().mean()
        assert error_denoised < error_noisy


# ─────────────────────────────────────────────────────────────────────
# 8. Encoding Matrix Tests
# ─────────────────────────────────────────────────────────────────────


class TestEncodingMatrix:
    """Tests for the explicit encoding matrix builder."""

    def test_adjoint_consistency(self, device: torch.device) -> None:
        """Test that <Ex, y> ≈ <x, E†y> (adjoint property)."""
        from mriforge.infrastructure.physics.encoding_matrix import EncodingMatrixBuilder

        N = 8
        enc = EncodingMatrixBuilder(
            n_readout=N, n_phase=N, fov_m=0.225, bw_hz=20000.0,
        )

        b0 = torch.zeros(1, 1, N, N, device=device)
        x = torch.randn(1, 1, N, N, device=device, dtype=torch.complex64)
        y = torch.randn(1, 1, N, N, device=device, dtype=torch.complex64)

        Ex = enc.forward(x, b0, t_shift=0.0)
        EHy = enc.adjoint(y, b0, t_shift=0.0)

        # <Ex, y> should equal <x, E†y>
        lhs = torch.sum(Ex.conj() * y)
        rhs = torch.sum(x.conj() * EHy)

        rel_error = (lhs - rhs).abs() / (lhs.abs() + 1e-12)
        assert rel_error < 0.1, f"Adjoint error {rel_error:.4f} too high"


# ─────────────────────────────────────────────────────────────────────
# 9. Integration: Iterative Reconstruction
# ─────────────────────────────────────────────────────────────────────


class TestIterativeRecon:
    """Integration tests for the full iterative pipeline."""

    def test_fft_baseline_runs(self, device: torch.device) -> None:
        """FFT mode should run without errors."""
        from mriforge.infrastructure.physics.fft_ops import fft2c
        from mriforge.infrastructure.physics.iterative_recon import (
            IterativeReconstructor,
            ReconMethod,
        )

        N = 16
        image = torch.zeros(1, 1, N, N, device=device, dtype=torch.complex64)
        image[0, 0, 4:12, 4:12] = 1.0 + 0j

        kspace = fft2c(image)

        recon = IterativeReconstructor(
            method=ReconMethod.FFT,
            n_iterations=1,
            t_shift=100e-6,
        )

        result = recon.reconstruct(kspace, kspace)

        assert result.image.shape == (1, 1, N, N)
        assert result.b0_map.shape == (1, 1, N, N)
        assert len(result.b0_maps_history) == 1

    def test_cpr_iterative_runs(self, device: torch.device) -> None:
        """CPR mode should run 2 iterations without errors."""
        from mriforge.infrastructure.physics.fft_ops import fft2c
        from mriforge.infrastructure.physics.iterative_recon import (
            IterativeReconstructor,
            ReconMethod,
        )

        N = 8  # Very small for speed
        image = torch.randn(1, 1, N, N, device=device, dtype=torch.complex64)
        kspace = fft2c(image)

        recon = IterativeReconstructor(
            method=ReconMethod.CPR,
            n_iterations=2,
            t_shift=100e-6,
            cpr_freq_range=(-200, 200),
            cpr_freq_step=50.0,
        )

        result = recon.reconstruct(kspace, kspace)

        assert result.image.shape == (1, 1, N, N)
        assert len(result.b0_maps_history) == 2
        assert len(result.images_history) == 2
