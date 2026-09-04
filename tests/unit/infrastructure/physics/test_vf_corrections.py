"""Unit tests for dual-marker correction operators (vf_corrections.py)."""

import torch

from spectramr.infrastructure.physics.vf_corrections import (
    AnalyticalPSFSolver,
    BiasFieldCorrector,
    EPINyquistGhostCorrector,
    GNLUnwarpingOperator,
    KSpacePhaseAutofocus,
    RespiratoryBinningAverager,
    SENSEGFactorCalibrator,
    SubVoxelSuperResolution,
)

# ────────────────────────────────────────────────────────────────────
# 1. GNLUnwarpingOperator
# ────────────────────────────────────────────────────────────────────


class TestGNLUnwarpingOperator:
    """Tests for gradient non-linearity unwarping."""

    def test_zero_flow_is_identity(self) -> None:
        gnl = GNLUnwarpingOperator(im_size=(16, 16))
        img = torch.randn(2, 1, 16, 16)
        flow = torch.zeros(1, 2, 16, 16)
        out = gnl(img, flow)
        # With zero displacement, Jacobian det = 1 everywhere
        inner = slice(2, 14)
        assert torch.allclose(
            out[:, :, inner, inner], img[:, :, inner, inner], atol=1e-4
        )

    def test_shape_preserved(self) -> None:
        gnl = GNLUnwarpingOperator(im_size=(32, 32))
        out = gnl(torch.randn(2, 1, 32, 32), torch.zeros(1, 2, 32, 32))
        assert out.shape == (2, 1, 32, 32)

    def test_jacobian_modulates_intensity(self) -> None:
        """Non-zero flow should produce Jacobian != 1 somewhere."""
        gnl = GNLUnwarpingOperator(im_size=(16, 16))
        flow = torch.randn(1, 2, 16, 16) * 0.1
        jac = gnl.compute_jacobian_det(flow)
        assert jac.shape == (1, 1, 16, 16)
        assert not torch.allclose(jac, torch.ones_like(jac), atol=1e-3)

    def test_estimate_deformation(self) -> None:
        gnl = GNLUnwarpingOperator(im_size=(16, 16), poly_order=2)
        true_pos = torch.tensor([[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]])
        dist_pos = true_pos + 0.05 * torch.randn(4, 2)
        flow = gnl.estimate_deformation(true_pos, dist_pos)
        assert flow.shape == (1, 2, 16, 16)


# ────────────────────────────────────────────────────────────────────
# 2. BiasFieldCorrector
# ────────────────────────────────────────────────────────────────────


class TestBiasFieldCorrector:
    """Tests for B1 bias field correction."""

    def test_uniform_bias_passthrough(self) -> None:
        bf = BiasFieldCorrector(smooth_sigma=3.0)
        img = torch.randn(2, 1, 16, 16)
        uniform_bias = torch.ones(2, 1, 16, 16)
        out = bf(img, uniform_bias)
        # Uniform bias → correction should be close to identity
        assert torch.allclose(out, img, atol=0.1)

    def test_shape_preserved(self) -> None:
        bf = BiasFieldCorrector(smooth_sigma=3.0)
        out = bf(torch.randn(2, 3, 32, 32), torch.ones(2, 1, 32, 32) * 1.5)
        assert out.shape == (2, 3, 32, 32)


# ────────────────────────────────────────────────────────────────────
# 3. AnalyticalPSFSolver
# ────────────────────────────────────────────────────────────────────


class TestAnalyticalPSFSolver:
    """Tests for Fourier-division PSF estimation."""

    def test_perfect_cross_gives_delta_psf(self) -> None:
        """Ideal cross (no blur) → PSF should be near delta."""
        solver = AnalyticalPSFSolver(cross_size=8, cross_width=2, wiener_lambda=0.01)
        H, W = 32, 32
        # Build ideal cross = observation (no blur)
        cross = solver._build_cross_prior(
            H, W, center=(16, 16), device=torch.device("cpu")
        )
        psf = solver.estimate_psf(cross, cross_center=(16, 16))
        assert psf.shape == (1, 1, H, W)
        # PSF should be concentrated at center
        assert psf[0, 0, 0, 0].abs() > 0.5  # DC component dominant

    def test_deconvolve_shape(self) -> None:
        solver = AnalyticalPSFSolver()
        psf = torch.zeros(1, 1, 16, 16)
        psf[0, 0, 0, 0] = 1.0  # Delta PSF
        out = solver.deconvolve(torch.randn(2, 1, 16, 16), psf)
        assert out.shape == (2, 1, 16, 16)

    def test_cross_prior_shape(self) -> None:
        solver = AnalyticalPSFSolver(cross_size=10, cross_width=3)
        cross = solver._build_cross_prior(
            32, 32, center=(16, 16), device=torch.device("cpu")
        )
        assert cross.shape == (1, 1, 32, 32)
        assert cross.sum() > 0  # Not all zeros


# ────────────────────────────────────────────────────────────────────
# 4. EPINyquistGhostCorrector
# ────────────────────────────────────────────────────────────────────


class TestEPINyquistGhostCorrector:
    """Tests for EPI N/2 ghost correction."""

    def test_shape_preserved(self) -> None:
        epi = EPINyquistGhostCorrector(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        phi_0 = torch.zeros(2)
        phi_1 = torch.zeros(2)
        out = epi(kspace, phi_0, phi_1)
        assert out.shape == (2, 1, 16, 16)

    def test_zero_phase_error_is_identity(self) -> None:
        epi = EPINyquistGhostCorrector(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        out = epi(kspace, torch.zeros(2), torch.zeros(2))
        assert torch.allclose(out, kspace, atol=1e-6)

    def test_correction_applies_to_odd_pe_lines_only(self) -> None:
        """The phase correction touches the ``1::2`` (odd-indexed) PE lines only;
        even lines pass through unchanged. Pins the behavior the inline comments
        used to contradict (they said "even")."""
        epi = EPINyquistGhostCorrector(im_size=(16, 16))
        kspace = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        out = epi(kspace, phi_0=torch.tensor([0.7]), phi_1=torch.tensor([0.3]))
        # Even PE lines are untouched...
        assert torch.allclose(out[:, :, 0::2, :], kspace[:, :, 0::2, :], atol=1e-6)
        # ...odd PE lines are modified by the non-zero phase correction.
        assert not torch.allclose(out[:, :, 1::2, :], kspace[:, :, 1::2, :], atol=1e-4)

    def test_estimate_phase_errors(self) -> None:
        epi = EPINyquistGhostCorrector(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1.0
        phi0, phi1 = epi.estimate_phase_errors(kspace, mask)
        assert phi0.shape == (2,) or phi0.ndim == 0
        assert phi1.shape == (2,) or phi1.ndim == 0

    def _make_marker_image(self, H, W, ghost_phase):
        """Image with a primary marker and its FOV/2 ghost (given readout phase)."""
        marker = torch.zeros(1, 1, H, W)
        marker[:, :, 2:5, 4:8] = 1.0
        img = torch.zeros(1, 1, H, W, dtype=torch.complex64)
        img[:, :, 2:5, 4:8] = 1.0  # primary
        img[:, :, 2 + H // 2 : 5 + H // 2, 4:8] = ghost_phase  # ghost at +FOV/2
        return marker, torch.fft.fft2(img, dim=(-2, -1))

    def test_phi1_estimated_nonzero_for_readout_phase_ramp(self) -> None:
        """phi_1 (the dominant ghost term) must be estimated, not hardcoded 0."""
        H = W = 16
        epi = EPINyquistGhostCorrector(im_size=(H, W))
        cols = torch.arange(W).float()
        ramp = torch.exp(1j * 0.4 * (cols - W / 2)).view(1, 1, 1, W)[..., 4:8]
        marker, kspace = self._make_marker_image(H, W, ramp)
        _, phi1 = epi.estimate_phase_errors(kspace, marker)
        assert phi1.abs().item() > 0.5  # genuinely non-zero (was always 0)

    def test_phi1_near_zero_for_constant_ghost(self) -> None:
        H = W = 16
        epi = EPINyquistGhostCorrector(im_size=(H, W))
        marker, kspace = self._make_marker_image(H, W, 1.0)  # no readout ramp
        _, phi1 = epi.estimate_phase_errors(kspace, marker)
        assert phi1.abs().item() < 0.1


# ────────────────────────────────────────────────────────────────────
# 5. RespiratoryBinningAverager
# ────────────────────────────────────────────────────────────────────


class TestRespiratoryBinningAverager:
    """Tests for respiratory binning and coherent averaging."""

    def test_average_shape(self) -> None:
        resp = RespiratoryBinningAverager(num_bins=4)
        frames = torch.randn(4, 1, 16, 16)
        avg = resp(frames)
        assert avg.shape == (1, 1, 16, 16)

    def test_averaging_reduces_noise(self) -> None:
        """Averaging N iid frames should reduce variance by factor N."""
        resp = RespiratoryBinningAverager(num_bins=8)
        frames = torch.randn(8, 1, 32, 32)
        avg = resp(frames)
        # Variance of average should be ~1/8 of single frame
        assert avg.var() < frames[0:1].var()

    def test_binning_assigns_all_lines(self) -> None:
        resp = RespiratoryBinningAverager(num_bins=4)
        trace = torch.rand(100)
        klines = torch.randn(100, 32)
        bins = resp.bin_by_phase(klines, trace)
        total_assigned = sum(len(b) for b in bins)
        assert total_assigned == 100


# ────────────────────────────────────────────────────────────────────
# 6. SubVoxelSuperResolution
# ────────────────────────────────────────────────────────────────────


class TestSubVoxelSuperResolution:
    """Tests for multi-frame sub-pixel super-resolution."""

    def test_output_shape(self) -> None:
        sr = SubVoxelSuperResolution(upscale_factor=2, num_iters=3)
        lr = torch.randn(4, 1, 8, 8)
        shifts = torch.randn(4, 2) * 0.5
        hr = sr(lr, shifts)
        assert hr.shape == (1, 1, 16, 16)

    def test_upscale_factor_3(self) -> None:
        sr = SubVoxelSuperResolution(upscale_factor=3, num_iters=2)
        lr = torch.randn(3, 1, 10, 10)
        shifts = torch.zeros(3, 2)
        hr = sr(lr, shifts)
        assert hr.shape == (1, 1, 30, 30)

    def test_single_frame(self) -> None:
        sr = SubVoxelSuperResolution(upscale_factor=2, num_iters=2)
        lr = torch.randn(1, 1, 8, 8)
        shifts = torch.zeros(1, 2)
        hr = sr(lr, shifts)
        assert hr.shape == (1, 1, 16, 16)

    def test_solver_builds_no_autograd_graph(self) -> None:
        """The manual-gradient solver must not retain an autograd graph: the
        result is detached and no grad_fn hangs off the iterate (the old
        ``x.requires_grad_(True)`` built a ~num_iters-deep graph for nothing)."""
        sr = SubVoxelSuperResolution(upscale_factor=2, num_iters=5, tv_lambda=0.01)
        lr = torch.randn(4, 1, 8, 8)
        shifts = torch.randn(4, 2) * 0.5
        hr = sr(lr, shifts)
        assert not hr.requires_grad
        assert hr.grad_fn is None

    def test_solver_reduces_data_fidelity(self) -> None:
        """Sanity: the iterate is a real solve, not a no-op — after iterating,
        the HR estimate downsamples closer to the observed frames than the
        bilinear-upsampled initialisation does."""
        import torch.nn.functional as F

        torch.manual_seed(0)
        sr = SubVoxelSuperResolution(upscale_factor=2, num_iters=20, tv_lambda=0.0)
        lr = torch.randn(3, 1, 8, 8)
        shifts = torch.zeros(3, 2)
        hr = sr(lr, shifts)
        init = F.interpolate(lr[0:1], size=(16, 16), mode="bilinear", align_corners=True)

        def fidelity(est: torch.Tensor) -> float:
            total = 0.0
            for n in range(lr.shape[0]):
                sim = sr._shift_and_downsample(est, (shifts * sr.upscale_factor)[n])
                total += float(((sim - lr[n : n + 1]) ** 2).mean())
            return total

        assert fidelity(hr) < fidelity(init)


# ────────────────────────────────────────────────────────────────────
# 7. KSpacePhaseAutofocus (alias check)
# ────────────────────────────────────────────────────────────────────


class TestKSpacePhaseAutofocus:
    """Tests for KSpacePhaseAutofocus alias."""

    def test_alias_is_rigid_kinematic(self) -> None:
        from spectramr.infrastructure.physics.vf_operators import RigidKinematicOperator

        assert KSpacePhaseAutofocus is RigidKinematicOperator

    def test_instantiation(self) -> None:
        af = KSpacePhaseAutofocus(im_size=(16, 16))
        assert af is not None


# ────────────────────────────────────────────────────────────────────
# SENSEGFactorCalibrator (vectorized g-factor)
# ────────────────────────────────────────────────────────────────────


class TestSENSEGFactorCalibrator:
    """Tests for the vectorized SENSE g-factor map."""

    def test_shape_and_lower_bound(self) -> None:
        cal = SENSEGFactorCalibrator(num_coils=4, acceleration=2)
        smaps = torch.complex(torch.randn(4, 16, 8), torch.randn(4, 16, 8))
        g = cal.compute_g_factor(smaps)
        assert g.shape == (1, 1, 16, 8)
        assert (g >= 1.0).all()  # g-factor >= 1 by definition

    def test_vectorized_matches_explicit_formula(self) -> None:
        """The batched path must equal the per-group g = sqrt((E^-1)_jj E_jj)."""
        torch.manual_seed(0)
        C, H, W, R = 4, 8, 4, 2
        smaps = torch.complex(torch.randn(C, H, W), torch.randn(C, H, W))
        cal = SENSEGFactorCalibrator(num_coils=C, acceleration=R)
        g = cal.compute_g_factor(smaps)

        gnum = H // R
        psi_inv = torch.eye(C, dtype=smaps.dtype)
        for offset in range(gnum):
            idx = [offset + r * gnum for r in range(R)]
            for w in range(W):
                s = torch.stack([smaps[:, i, w] for i in idx], dim=1)  # [C, R]
                e = s.conj().T @ psi_inv @ s
                e_inv = torch.linalg.inv(e)
                for r, pix in enumerate(idx):
                    ref = torch.sqrt(
                        (e_inv[r, r] * e[r, r]).real.abs()
                    ).clamp(min=1.0)
                    assert torch.allclose(g[0, 0, pix, w], ref, atol=1e-4)
