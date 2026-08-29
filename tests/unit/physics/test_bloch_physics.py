import numpy as np
import pytest
import torch

from mriforge.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer


class TestBlochPhysics:
    """
    Unit tests for the phase-sensitive SPGR Bloch simulation.
    Verifies B0 phase accumulation, B1 flip angle scaling, and complex invariance.
    """

    @pytest.fixture
    def bloch_layer(self):
        return DifferentiableBlochLayer(sequence_type="SPGR")

    def test_b0_phase_accumulation(self, bloch_layer):
        """
        Verify that B0 off-resonance correctly accumulates phase: φ = 2π * f * TE.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bloch_layer = bloch_layer.to(device)

        # Parameters
        t1 = torch.tensor([[[[1000.0]]]], device=device)
        t2 = torch.tensor([[[[100.0]]]], device=device)
        pd = torch.tensor([[[[1.0]]]], device=device)
        b0 = torch.tensor([[[[100.0]]]], device=device)  # 100 Hz
        tr, te = 10.0, 5.0  # ms
        flip = 0.5  # rad

        # Simulate
        signal = bloch_layer(t1, t2, pd, tr, te, flip, b0_map=b0)

        # Expected phase: 2 * pi * 100 Hz * 0.005 s = pi
        # Signal should be purely negative real (or close to it)
        # Use cos(phase) for robustness against +/- pi wrap
        phase = torch.angle(signal).item()
        assert np.allclose(np.cos(phase), -1.0, atol=1e-5)
        assert signal.real.item() < 0
        assert abs(signal.imag.item()) < 1e-6

    def test_b1_flip_angle_scaling(self, bloch_layer):
        """
        Verify that B1 correctly scales the effective flip angle.
        Signal at flip=0.5, B1=2.0 should match flip=1.0, B1=1.0.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bloch_layer = bloch_layer.to(device)

        t1 = torch.tensor([[[[1000.0]]]], device=device)
        t2 = torch.tensor([[[[100.0]]]], device=device)
        pd = torch.tensor([[[[1.0]]]], device=device)
        tr, te = 10.0, 1.0

        # Case A: Nominal flip 1.0, B1=1.0
        sig_a = bloch_layer(t1, t2, pd, tr, te, 1.0)

        # Case B: Nominal flip 0.5, B1=2.0
        b1 = torch.tensor([[[[2.0]]]], device=device)
        sig_b = bloch_layer(t1, t2, pd, tr, te, 0.5, b1_map=b1)

        assert torch.allclose(sig_a, sig_b, atol=1e-6)

    def test_complex_magnitude_invariant(self, bloch_layer):
        """
        Ensure magnitude of complex simulation (with B0) matches real simulation (without B0).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bloch_layer = bloch_layer.to(device)

        t1 = torch.randn(2, 1, 16, 16, device=device).abs() * 1000 + 10
        t2 = torch.randn(2, 1, 16, 16, device=device).abs() * 100 + 1
        pd = torch.randn(2, 1, 16, 16, device=device).abs()
        b0 = torch.randn(2, 1, 16, 16, device=device) * 200
        tr, te, flip = 15.0, 4.0, 0.3

        sig_real = bloch_layer(t1, t2, pd, tr, te, flip)
        sig_complex = bloch_layer(t1, t2, pd, tr, te, flip, b0_map=b0)

        assert torch.allclose(sig_real, sig_complex.abs(), atol=1e-6)

    def test_softplus_safety(self, bloch_layer):
        """
        Verify that negative T1/T2 inputs are handled safely via softplus.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bloch_layer = bloch_layer.to(device)

        t1 = torch.tensor([[[[-1000.0]]]], device=device)  # invalid
        t2 = torch.tensor([[[[-100.0]]]], device=device)  # invalid
        pd = torch.tensor([[[[1.0]]]], device=device)

        # Should not crash and should produce valid finite signal
        sig = bloch_layer(t1, t2, pd, 10.0, 2.0, 0.5)
        assert torch.isfinite(sig).all()
        assert (sig >= 0).all()
