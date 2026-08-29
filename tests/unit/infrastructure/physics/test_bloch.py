import math

import torch

from mriforge.infrastructure.physics.biophysics.bloch import BlochSimulator


class TestBlochSimulator:
    """Unit tests for biophysics.bloch.BlochSimulator."""

    @property
    def device(self):
        return torch.device("cpu")

    def test_spin_echo_basic(self):
        """Test Spin Echo signal equation basics."""
        # Case 1: Infinite TR, Short TE -> Signal approx M0
        m0 = torch.tensor([1.0], device=self.device)
        t1 = torch.tensor([1000.0], device=self.device)
        t2 = torch.tensor([100.0], device=self.device)
        te = 0.001  # very short
        tr = 1e6  # very long

        signal = BlochSimulator.spin_echo(m0, t1, t2, te, tr)
        assert torch.isclose(signal, m0, atol=1e-3)

        # Case 2: T2 decay
        # TE = T2 -> Signal drops by 1/e
        te = 100.0
        signal = BlochSimulator.spin_echo(m0, t1, t2, te, tr)
        expected = m0 * math.exp(-1.0)
        assert torch.isclose(signal, expected, atol=1e-3)

    def test_inversion_recovery_null_point(self):
        """Test Inversion Recovery null point."""
        # Null point happens at TI = T1 * ln(2) approx 0.693 * T1
        # Signal should be 0 there (if TR >> T1)
        m0 = torch.tensor([1.0], device=self.device)
        t1 = torch.tensor([1000.0], device=self.device)
        tr = 10000.0  # Long TR

        ti_null = 1000.0 * math.log(2.0)

        signal = BlochSimulator.inversion_recovery(m0, t1, ti_null, tr)

        # Should be very close to 0
        assert torch.allclose(signal, torch.zeros_like(signal), atol=1e-3)

    def test_gradient_echo_ernst_angle(self):
        """Test Gradient Echo signal maximizes at Ernst Angle."""
        # Ernst Angle: cos(alpha) = exp(-TR/T1)
        m0 = torch.tensor([1.0])
        t1 = torch.tensor([1000.0])
        t2_star = torch.tensor([100.0])
        tr = 100.0
        te = 0.0  # Ignore T2* for amplitude check

        # Calculate Ernst Angle
        e1 = math.exp(-tr / t1.item())
        ernst_rad = math.acos(e1)

        signal_ernst = BlochSimulator.gradient_echo(m0, t1, t2_star, te, tr, ernst_rad)

        # Check nearby angles (slightly less and slightly more)
        signal_less = BlochSimulator.gradient_echo(
            m0, t1, t2_star, te, tr, ernst_rad - 0.1
        )
        signal_more = BlochSimulator.gradient_echo(
            m0, t1, t2_star, te, tr, ernst_rad + 0.1
        )

        assert signal_ernst > signal_less
        assert signal_ernst > signal_more

    def test_bloch_step_matrix_exp_relaxation(self):
        """Test Matrix Exponential solver for relaxation."""
        # Initial M = [1, 0, 0] (Transverse)
        m_init = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)  # [1, 3]
        t1 = torch.tensor([1000.0], device=self.device)
        t2 = torch.tensor([100.0], device=self.device)

        # Step 1: Wait for T2. Transverse should decay to 1/e (~0.368)
        dt = 100.0
        m_new = BlochSimulator.bloch_step_matrix_exp(
            m_init, t1, t2, dt, df=0, b1_phase=0, flip_angle=0
        )

        # Mx should be exp(-1), My 0, Mz recovering from 0 towards 1
        expected_mx = math.exp(-1.0)
        assert torch.isclose(m_new[0, 0], torch.tensor(expected_mx), atol=1e-3)

        # Step 2: Wait very long. Mz should recover to 1.
        dt_long = 10000.0
        m_relaxed = BlochSimulator.bloch_step_matrix_exp(m_new, t1, t2, dt_long)
        assert torch.isclose(m_relaxed[0, 2], torch.tensor(1.0), atol=1e-3)

    def test_bloch_step_matrix_exp_precession(self):
        """Test Larmor precession."""
        # Initial Mx = 1
        m_init = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        t1 = torch.tensor([1e9])  # Ignore relaxation
        t2 = torch.tensor([1e9])

        # df = 100 Hz. dt = 2.5 ms (1/4 cycle).
        # Should rotate 90 degrees in transverse plane (to My or -My depending on sign convention)
        df = 100.0
        dt = 2.5  # 100 Hz * 0.0025 s = 0.25 cycles = 90 deg

        m_new = BlochSimulator.bloch_step_matrix_exp(m_init, t1, t2, dt, df=df)

        # Mx -> 0, My -> +/- 1
        assert torch.isclose(m_new[0, 0], torch.tensor(0.0), atol=1e-3)
        assert abs(m_new[0, 1]) > 0.99  # Magnitude 1

    def test_bloch_step_matrix_exp_b1_phase_selects_axis(self):
        """RF phase selects the transverse axis; b1_phase must not be a no-op.

        A 90-degree pulse on equilibrium M=(0,0,1): an x-phase pulse (b1_phase=0)
        tips Mz->-My; a y-phase pulse (b1_phase=pi/2) tips Mz->+Mx. b1_phase was
        previously computed but ignored (rotation hardcoded about x).
        """
        m = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        t1 = torch.tensor([1e9], device=self.device)  # ignore relaxation
        t2 = torch.tensor([1e9], device=self.device)
        dt = 1e-3
        fa = math.pi / 2

        m_x = BlochSimulator.bloch_step_matrix_exp(
            m, t1, t2, dt, b1_phase=0.0, flip_angle=fa
        )
        m_y = BlochSimulator.bloch_step_matrix_exp(
            m, t1, t2, dt, b1_phase=math.pi / 2, flip_angle=fa
        )

        assert torch.isclose(m_x[0, 1], torch.tensor(-1.0), atol=1e-2)  # -My
        assert torch.isclose(m_y[0, 0], torch.tensor(1.0), atol=1e-2)  # +Mx
        assert not torch.allclose(m_x, m_y, atol=1e-2)  # b1_phase is wired
