"""Integration tests for Pipeline A (Koopman-Advection) and Pipeline B (Quantum NeRF)."""

import pytest
import torch

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Forward Model Infrastructure
# ──────────────────────────────────────────────────────────────────────


class TestGoldenAngleTrajectory:
    """Tests for golden-angle radial trajectory generation."""

    def test_trajectory_shape(self):
        from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

        gen = GoldenAngleTrajectory(
            im_size=(64, 64), num_spokes=16, samples_per_spoke=32
        )
        traj = gen.generate(num_frames=5)
        assert traj.shape == (5, 2, 16 * 32)

    def test_trajectory_range(self):
        """Coordinates must be in [-π, π]."""
        from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

        gen = GoldenAngleTrajectory(
            im_size=(64, 64), num_spokes=8, samples_per_spoke=16
        )
        traj = gen.generate(num_frames=1)
        assert traj.abs().max() <= 3.15  # ≈ π

    def test_golden_angle_spacing(self):
        """Consecutive spokes should be separated by the golden angle."""
        import math

        from spectramr.infrastructure.physics.nufft_ops import GOLDEN_ANGLE

        assert abs(GOLDEN_ANGLE - math.pi * (math.sqrt(5) - 1) / 2) < 1e-10


class TestNUFFTForwardModel:
    """Tests for NUFFT forward/adjoint."""

    @pytest.fixture
    def nufft(self):
        from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel

        return NUFFTForwardModel(im_size=(32, 32))

    def test_forward_shape(self, nufft):
        from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

        gen = GoldenAngleTrajectory(
            im_size=(32, 32), num_spokes=8, samples_per_spoke=32
        )
        traj = gen.generate(1)[0]  # [2, N]
        img = torch.complex(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        kdata = nufft.forward_project(img, traj)
        assert kdata.shape == (1, 1, 8 * 32)

    def test_adjoint_shape(self, nufft):
        from spectramr.infrastructure.physics.nufft_ops import GoldenAngleTrajectory

        gen = GoldenAngleTrajectory(
            im_size=(32, 32), num_spokes=8, samples_per_spoke=32
        )
        traj = gen.generate(1)[0]
        kdata = torch.complex(torch.randn(1, 1, 256), torch.randn(1, 1, 256))
        img = nufft.adjoint_project(kdata, traj)
        assert img.shape == (1, 1, 32, 32)


class TestSPAMMGrid:
    """Tests for SPAMM injection and Dirac notch."""

    def test_spamm_modulates_magnitude(self):
        from spectramr.infrastructure.physics.spamm_grid import SPAMMGridInjector

        spamm = SPAMMGridInjector(spatial_freq_x=4.0, spatial_freq_y=4.0)
        img = torch.ones(1, 1, 64, 64)
        tagged = spamm(img)
        # Tagged image should have variation (not all ones)
        assert tagged.std() > 0.01

    def test_dirac_notch_coverage(self):
        """Notch filter should block only a tiny fraction of k-space."""
        from spectramr.infrastructure.physics.spamm_grid import DiracNotchFilter

        notch = DiracNotchFilter(
            im_size=(64, 64), spatial_freq_x=8.0, spatial_freq_y=8.0
        )
        mask = notch.get_loss_mask()
        coverage = mask.sum() / mask.numel()
        assert coverage > 0.95  # <5% blocked

    def test_notch_zeros_correct_positions(self):
        """Notch filter zeros should be at SPAMM frequency positions."""
        from spectramr.infrastructure.physics.spamm_grid import DiracNotchFilter

        notch = DiracNotchFilter(
            im_size=(64, 64), spatial_freq_x=8.0, spatial_freq_y=8.0, notch_radius=2
        )
        mask = notch.get_loss_mask()
        # Center of notch at (32+8, 32+8) = (40, 40) should be 0
        assert mask[40, 40] == 0.0

