"""Unit tests for the final 7 framework components.

Tests cover:
- 3 new field extractors (BlochSiegert, SSFP Banding, STEAM/SE)
- 2 new ML modules (DictionaryLearner, SpatiotemporalAttention)
- 2 new operators (RFModulation, NUFFTTrajectory)
"""

import math

import torch

from mriforge.infrastructure.physics.vf_field_extraction import (
    BlochSiegertB1Estimator,
    SSFPBandingB0Estimator,
    STEAMSpinEchoB1Estimator,
)
from mriforge.infrastructure.physics.vf_operators_extended import (
    NUFFTTrajectoryCalibrator,
    RFModulationOperator,
)
from mriforge.models.blocks.vf_modules import (
    MarkerDictionaryLearner,
    SpatiotemporalMarkerAttention,
)

# ────────────────────────────────────────────────────────────────────
# 1. BlochSiegertB1Estimator
# ────────────────────────────────────────────────────────────────────


class TestBlochSiegertB1Estimator:
    """Tests for Bloch-Siegert B1+ estimation."""

    def test_shape(self) -> None:
        bs = BlochSiegertB1Estimator()
        s_pos = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        s_neg = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        out = bs(s_pos, s_neg)
        assert out.shape == (2, 1, 16, 16)

    def test_identical_signals_give_zero(self) -> None:
        """Same ±ω signals → phase_diff = 0 → B1+ = 0."""
        bs = BlochSiegertB1Estimator()
        sig = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        out = bs(sig, sig)
        # B1 scale from zero phase should be near zero (before normalisation)
        assert out.abs().max() < 0.1

    def test_nonzero_for_phase_difference(self) -> None:
        bs = BlochSiegertB1Estimator()
        s_pos = torch.complex(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8) * 0.5)
        s_neg = torch.complex(torch.ones(1, 1, 8, 8), -torch.ones(1, 1, 8, 8) * 0.5)
        out = bs(s_pos, s_neg)
        assert out.abs().max() > 0


# ────────────────────────────────────────────────────────────────────
# 2. SSFPBandingB0Estimator
# ────────────────────────────────────────────────────────────────────


class TestSSFPBandingB0Estimator:
    """Tests for bSSFP banding B0 estimation."""

    def test_shape(self) -> None:
        ssfp = SSFPBandingB0Estimator(tr_s=0.005)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1
        out = ssfp(torch.randn(2, 1, 16, 16).abs(), mask)
        assert out.shape == (2, 1, 16, 16)

    def test_output_masked(self) -> None:
        """Output must be zero outside marker mask."""
        ssfp = SSFPBandingB0Estimator()
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 6:10, 6:10] = 1
        out = ssfp(torch.randn(1, 1, 16, 16).abs(), mask)
        # Outside mask must be zero
        outside = out * (1 - mask)
        assert outside.abs().sum() < 1e-6

    def test_band_count_halved_and_normalized_by_extent(self) -> None:
        """Tier C fix: gradient = (transitions/2) / (TR · marker_extent), NOT
        transitions / (TR · H). Full-height marker isolates the arithmetic."""
        tr, H, W = 0.005, 16, 16
        ssfp = SSFPBandingB0Estimator(tr_s=tr)
        mask = torch.ones(1, 1, H, W)  # extent = H, no non-marker pollution
        img = torch.ones(1, 1, H, W)
        for r in (4, 8, 12):  # 3 isolated dark bands -> 6 threshold crossings
            img[:, :, r, :] = 0.0
        out = ssfp(img, mask)
        expected = 3.0 / (tr * H)  # n_bands=3, extent=H -> 37.5 Hz/pixel
        val = float(out[0, 0, 0, 0])
        assert abs(val - expected) < 1e-3
        # Exactly half of the old raw-transition-count form (the ÷2 fix).
        raw_form = 6.0 / (tr * H)
        assert abs(val - raw_form / 2.0) < 1e-3


# ────────────────────────────────────────────────────────────────────
# 3. STEAMSpinEchoB1Estimator
# ────────────────────────────────────────────────────────────────────


class TestSTEAMSpinEchoB1Estimator:
    """Tests for STEAM/SE ratio B1+ estimation."""

    def test_shape(self) -> None:
        est = STEAMSpinEchoB1Estimator(nominal_flip_deg=90)
        out = est(torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16) + 0.1)
        assert out.shape == (2, 1, 16, 16)

    def test_perfect_90_deg_gives_unity(self) -> None:
        """For nominal 90° pulse with sin²(45°) = 0.5 ratio → B1+ = 1."""
        est = STEAMSpinEchoB1Estimator(nominal_flip_deg=90)
        ratio_ideal = math.sin(math.radians(45)) ** 2  # 0.5
        s_steam = torch.ones(1, 1, 8, 8) * ratio_ideal
        s_se = torch.ones(1, 1, 8, 8)
        out = est(s_steam, s_se)
        assert torch.allclose(out, torch.ones_like(out), atol=0.01)


# ────────────────────────────────────────────────────────────────────
# 4. MarkerDictionaryLearner
# ────────────────────────────────────────────────────────────────────


class TestMarkerDictionaryLearner:
    """Tests for K-SVD sparse coding dictionary learner."""

    def test_shape(self) -> None:
        dl = MarkerDictionaryLearner(patch_size=4, num_atoms=16, sparsity=2)
        out = dl(torch.randn(1, 1, 16, 16))
        assert out.shape == (1, 1, 16, 16)

    def test_artifact_identification(self) -> None:
        """Identify_artifact_atoms should return boolean mask of correct size."""
        dl = MarkerDictionaryLearner(patch_size=4, num_atoms=8, sparsity=2)
        corrupted = torch.randn(1, 1, 16, 16)
        ideal = torch.randn(1, 1, 16, 16)
        mask = dl.identify_artifact_atoms(corrupted, ideal)
        assert mask.shape == (8,)
        assert mask.dtype == torch.bool

    def test_patch_extraction_roundtrip(self) -> None:
        dl = MarkerDictionaryLearner(patch_size=4, num_atoms=64, sparsity=4)
        img = torch.randn(1, 1, 16, 16)
        patches = dl._extract_patches(img)
        assert patches.shape == (16, 16)  # 4×4 patches from 16×16 image


# ────────────────────────────────────────────────────────────────────
# 5. SpatiotemporalMarkerAttention
# ────────────────────────────────────────────────────────────────────


class TestSpatiotemporalMarkerAttention:
    """Tests for cross-attention temporal noise subtraction."""

    def test_shape(self) -> None:
        sta = SpatiotemporalMarkerAttention(dim=32, num_heads=2, num_timepoints=8)
        out = sta(torch.randn(2, 1, 16, 16), torch.randn(2, 8))
        assert out.shape == (2, 1, 16, 16)

    def test_gradient_flow(self) -> None:
        sta = SpatiotemporalMarkerAttention(dim=32, num_heads=2, num_timepoints=8)
        x = torch.randn(1, 1, 8, 8, requires_grad=True)
        m = torch.randn(1, 8)
        out = sta(x, m)
        out.sum().backward()
        assert x.grad is not None

    def test_3d_marker_temporal(self) -> None:
        """Should handle [B, T, H_m, W_m] marker input."""
        sta = SpatiotemporalMarkerAttention(dim=16, num_heads=2, num_timepoints=4)
        out = sta(torch.randn(1, 1, 8, 8), torch.randn(1, 4, 3, 3))
        assert out.shape == (1, 1, 8, 8)


# ────────────────────────────────────────────────────────────────────
# 6. RFModulationOperator
# ────────────────────────────────────────────────────────────────────


class TestRFModulationOperator:
    """Tests for RF modulation forward/adjoint."""

    def test_forward_shape(self) -> None:
        rf = RFModulationOperator(num_coils=1)
        out = rf(
            torch.randn(2, 1, 16, 16),
            torch.ones(2, 1, 16, 16),
            torch.ones(2, 1, 16, 16),
        )
        assert out.shape == (2, 1, 16, 16)

    def test_identity_passthrough(self) -> None:
        """B1+ = B1- = 1 → output = input."""
        rf = RFModulationOperator(num_coils=1)
        img = torch.randn(2, 1, 16, 16)
        out = rf(img, torch.ones(2, 1, 16, 16), torch.ones(2, 1, 16, 16))
        assert torch.allclose(out, img, atol=1e-6)

    def test_adjoint_shape(self) -> None:
        rf = RFModulationOperator(num_coils=4)
        coil_images = torch.randn(2, 4, 16, 16)
        b1p = torch.ones(2, 1, 16, 16)
        b1m = torch.ones(2, 4, 16, 16) * 0.5
        out = rf.adjoint(coil_images, b1p, b1m)
        assert out.shape == (2, 1, 16, 16)


# ────────────────────────────────────────────────────────────────────
# 7. NUFFTTrajectoryCalibrator
# ────────────────────────────────────────────────────────────────────


class TestNUFFTTrajectoryCalibrator:
    """Tests for NUFFT trajectory delay calibration."""

    def test_forward_shape(self) -> None:
        nufft = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        out = nufft(kspace)
        assert out.shape == (2, 1, 16, 16)

    def test_zero_delay_is_identity(self) -> None:
        nufft = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        kspace = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
        out = nufft(kspace)
        assert torch.allclose(out, kspace, atol=1e-6)

    def test_phase_correction_shape(self) -> None:
        nufft = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        correction = nufft.compute_phase_correction(torch.tensor([0.5, -0.3]))
        assert correction.shape == (1, 1, 16, 16)
        assert torch.is_complex(correction)

    def test_phase_correction_dc_is_centered(self) -> None:
        """The DC column (W//2) must carry zero phase for a readout-only delay.

        Regression for the k-grid convention: fftshift(fftfreq) puts an exact 0
        at index N//2, so the ramp has unit phasor at DC; linspace(-0.5,0.5,N)
        has no exact DC zero (for even N) and would leave a residual phase.
        """
        nufft = NUFFTTrajectoryCalibrator(im_size=(16, 16))
        corr = nufft.compute_phase_correction(torch.tensor([0.5, 0.0]))
        dc_col = corr[0, 0, :, 8]  # W // 2 == 8
        assert torch.allclose(dc_col, torch.ones_like(dc_col), atol=1e-5)
