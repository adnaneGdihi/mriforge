"""Unit tests for VF field extraction (vf_field_extraction.py)."""

import inspect
import math

import pytest
import torch

from mriforge.infrastructure.physics.vf_field_extraction import (
    K0PhaseDriftTracker,
    LarmorDriftTracker,
    MarkerB0FromDualEcho,
    MarkerB1ReceiveInversion,
    MarkerB1TransmitAFI,
    MarkerB1TransmitDoubleAngle,
    PhaseUnwrappingSeed,
    SusceptibilityB0Solver,
)


class TestMarkerB0FromDualEcho:
    """Tests for dual-echo B0 extraction."""

    def test_zero_phase_diff_gives_zero_b0(self) -> None:
        """Identical echoes → zero B0."""
        estimator = MarkerB0FromDualEcho(delta_te=0.00246)
        echo = torch.complex(torch.ones(1, 1, 16, 16), torch.zeros(1, 1, 16, 16))
        b0 = estimator(echo, echo)
        assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-5)

    def test_known_phase_diff(self) -> None:
        """Known phase difference → correct B0."""
        delta_te = 0.00246
        expected_b0 = 50.0  # Hz
        phase = 2.0 * math.pi * expected_b0 * delta_te
        echo1 = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        echo2 = torch.complex(
            torch.cos(torch.tensor(phase)) * torch.ones(1, 1, 8, 8),
            torch.sin(torch.tensor(phase)) * torch.ones(1, 1, 8, 8),
        )
        estimator = MarkerB0FromDualEcho(delta_te=delta_te)
        b0_map = estimator(echo1, echo2)
        assert torch.allclose(b0_map, torch.ones_like(b0_map) * expected_b0, atol=0.5)

    def test_mask_restricts_output(self) -> None:
        estimator = MarkerB0FromDualEcho()
        echo1 = torch.complex(torch.ones(1, 1, 8, 8), torch.zeros(1, 1, 8, 8))
        echo2 = echo1 * torch.exp(1j * torch.tensor(0.1))
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0:4, 0:4] = 1.0
        b0 = estimator(echo1, echo2, marker_mask=mask)
        assert torch.allclose(b0[:, :, 4:, :], torch.zeros(1, 1, 4, 8))

    def test_aliasing_limit_at_half_inverse_delta_te(self) -> None:
        """Within ±1/(2·ΔTE) the field is recovered; beyond it wraps (aliases).

        Documents the single-difference (no-unwrap) range limit: torch.angle
        wraps to (-π, π], so |ΔB0| < 1/(2·ΔTE) is the recoverable band.
        """
        dte = 0.00246
        est = MarkerB0FromDualEcho(delta_te=dte)
        limit = 1.0 / (2.0 * dte)  # ≈ 203 Hz

        def recover(b0_hz: float) -> float:
            e1 = torch.complex(torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
            e2 = e1 * torch.exp(1j * torch.tensor(2.0 * math.pi * b0_hz * dte))
            return est(e1, e2).mean().item()

        # Within the band: recovered ≈ true field.
        assert abs(recover(100.0) - 100.0) < 1.0
        # Beyond the band: aliases by 1/ΔTE, so recovered != true.
        assert abs(recover(limit + 100.0) - (limit + 100.0)) > 50.0

    def test_real_input_raises(self) -> None:
        """Real (non-complex) echoes must raise, not silently return zero B0.

        Regression for the silent-fallback pitfall (#9): a pure real magnitude
        image makes ``.conj()`` a no-op and ``angle()`` collapse to 0/π, so the
        old code returned a bogus all-zero B0 map indistinguishable from a true
        no-field result. The strict complex coercion now fails loud instead.
        """
        estimator = MarkerB0FromDualEcho(delta_te=0.00246)
        real_echo = torch.ones(1, 1, 8, 8)  # pure real magnitude, C=1 (odd)
        with pytest.raises(ValueError):
            estimator(real_echo, real_echo)


class TestMarkerB1TransmitDoubleAngle:
    """Tests for double-angle B1+ estimation."""

    def test_shape(self) -> None:
        estimator = MarkerB1TransmitDoubleAngle()
        s_alpha = torch.ones(1, 1, 8, 8)
        s_2alpha = 2.0 * torch.ones(1, 1, 8, 8) * 0.85  # cos(α) ≈ 0.85
        b1 = estimator(s_alpha, s_2alpha)
        assert b1.shape == (1, 1, 8, 8)

    def test_recovers_known_flip_angle(self) -> None:
        """The (previously dead) T1 correction exactly inverts ideal
        spoiled-GRE signals back to the true flip angle."""
        t1_marker, tr, nominal = 300.0, 50.0, 30.0
        e1 = math.exp(-tr / t1_marker)
        a_true = math.radians(25.0)

        def spgr(a: float) -> float:
            return math.sin(a) * (1 - e1) / (1 - e1 * math.cos(a))

        s_a = torch.full((1, 1, 4, 4), spgr(a_true))
        s_2a = torch.full((1, 1, 4, 4), spgr(2 * a_true))
        est = MarkerB1TransmitDoubleAngle(
            t1_marker=t1_marker, tr=tr, nominal_alpha=nominal
        )
        recovered = (est(s_a, s_2a) * est.nominal_alpha_rad).mean().item()
        assert recovered == pytest.approx(a_true, abs=1e-3)

    def test_reduces_to_uncorrected_in_long_tr_limit(self) -> None:
        """E1→0 (TR≫T1): correction collapses to the plain arccos(R/2)."""
        est = MarkerB1TransmitDoubleAngle(t1_marker=1.0, tr=50.0, nominal_alpha=30.0)
        s_a = torch.ones(1, 1, 4, 4)
        s_2a = 1.2 * torch.ones(1, 1, 4, 4)
        recovered = (est(s_a, s_2a) * est.nominal_alpha_rad).mean().item()
        assert recovered == pytest.approx(math.acos(1.2 / 2), abs=1e-4)


class TestMarkerB1TransmitAFI:
    """Tests for AFI B1+ estimation."""

    def test_shape(self) -> None:
        estimator = MarkerB1TransmitAFI()
        s1 = torch.ones(1, 1, 8, 8)
        s2 = 1.5 * torch.ones(1, 1, 8, 8)
        b1 = estimator(s1, s2)
        assert b1.shape == (1, 1, 8, 8)


class TestMarkerB1ReceiveInversion:
    """Tests for analytical B1- receive extraction."""

    def test_uniform_signal_gives_constant_b1(self) -> None:
        estimator = MarkerB1ReceiveInversion(
            t1=300.0, t2=50.0, rho=1.0, tr=50.0, te=10.0, flip_angle=15.0
        )
        signal = torch.ones(1, 1, 8, 8) * estimator.s_bloch  # Exact theoretical signal
        b1 = estimator(signal)
        assert torch.allclose(b1, torch.ones_like(b1), atol=1e-4)


class TestLarmorDriftTracker:
    """Tests for spectral drift tracking."""

    def test_zero_signal_returns_reference(self) -> None:
        tracker = LarmorDriftTracker(reference_freq_hz=0.0)
        # DC signal → peak at 0 Hz → drift = 0
        signal = torch.ones(100, dtype=torch.complex64)
        drift = tracker(signal, dt=0.001)
        assert drift.shape[0] == 1
        assert abs(drift.item()) < 1.0  # near zero


class TestSusceptibilityB0Solver:
    """Tests for susceptibility-based B0 estimation."""

    def test_zero_susceptibility_gives_zero_b0(self) -> None:
        solver = SusceptibilityB0Solver(b0_tesla=0.064)
        chi = torch.zeros(1, 1, 16, 16)
        b0 = solver(chi)
        assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-6)

    def test_shape(self) -> None:
        solver = SusceptibilityB0Solver()
        chi = torch.randn(2, 1, 32, 32)
        assert solver(chi).shape == (2, 1, 32, 32)

    def test_no_unwired_susceptibility_knobs(self) -> None:
        """Regression for pitfall #15: chi_marker/chi_air were advertised but
        never read in forward; the solver must not expose them or delta_chi."""
        params = inspect.signature(SusceptibilityB0Solver.__init__).parameters
        assert "chi_marker" not in params
        assert "chi_air" not in params
        solver = SusceptibilityB0Solver(b0_tesla=0.064)
        assert not hasattr(solver, "delta_chi")


class TestK0PhaseDriftTracker:
    """Tests for k-space centre phase drift tracking."""

    def test_shape(self) -> None:
        tracker = K0PhaseDriftTracker(te=0.01)
        kspace = torch.complex(torch.randn(5, 1, 16, 16), torch.randn(5, 1, 16, 16))
        drift = tracker(kspace)
        assert drift.shape == (5,)

    def test_first_timepoint_zero(self) -> None:
        """B0 drift at t=0 should be zero (reference point)."""
        tracker = K0PhaseDriftTracker(te=0.01)
        kspace = torch.complex(torch.randn(5, 1, 16, 16), torch.randn(5, 1, 16, 16))
        drift = tracker(kspace)
        assert drift[0].abs() < 1e-4

    def test_with_marker_mask(self) -> None:
        tracker = K0PhaseDriftTracker(te=0.01)
        kspace = torch.complex(torch.randn(3, 1, 16, 16), torch.randn(3, 1, 16, 16))
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1
        drift = tracker(kspace, marker_mask=mask)
        assert drift.shape == (3,)


class TestPhaseUnwrappingSeed:
    """Tests for marker-seeded phase unwrapping."""

    def test_shape(self) -> None:
        unwrapper = PhaseUnwrappingSeed(known_b0_at_marker=0.0, te=0.01)
        phase = torch.randn(2, 1, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1
        out = unwrapper(phase, mask)
        assert out.shape == (2, 1, 16, 16)

    def test_smooth_phase_unchanged(self) -> None:
        """Smooth phase (no wraps) should be approximately preserved."""
        unwrapper = PhaseUnwrappingSeed(known_b0_at_marker=0.0, te=0.01)
        smooth = torch.linspace(-1.0, 1.0, 16).unsqueeze(0).expand(16, -1)
        phase = smooth.unsqueeze(0).unsqueeze(0)  # [1,1,16,16]
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 0:4, 0:4] = 1
        out = unwrapper(phase, mask)
        # The smooth structure should be preserved (within 2π multiples)
        diff = (out - phase).abs()
        residual = diff - torch.round(diff / (2 * math.pi)) * 2 * math.pi
        assert residual.abs().max() < 0.5
