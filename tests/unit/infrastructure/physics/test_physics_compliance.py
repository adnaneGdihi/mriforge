import unittest

import numpy as np
import torch

from spectramr.infrastructure.physics.field_simulation import B0MapSimulator
from spectramr.infrastructure.physics.implementations.gaussian_fourier_op import (
    differentiable_gaussian_fourier,
)
from spectramr.infrastructure.physics.integration import BlochEquationSolver


class TestPhysicsCompliance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_bloch_t2_star_dephasing(self):
        """Verify that B0 inhomogeneity causes signal dephasing (T2* decay)."""
        # Create solver
        solver = BlochEquationSolver(
            t1=1000.0, t2=100.0, tr=500.0, te=50.0, voxel_size=1.0
        )

        # 1. Homogeneous field (d(B0)/dx = 0) -> No dephasing (magnitude should be exp(-te/t2))
        b0_map_flat = torch.zeros((1, 10, 10))
        # Note: forward takes b0_map
        m0 = torch.zeros((1, 10, 10, 3))
        m0[..., 2] = 1.0  # Equilibrium

        # Forward pass (Flip 90)
        signal_flat = solver.forward(m0, b0_map=b0_map_flat)
        mag_flat = torch.norm(signal_flat, dim=-1)

        expected_decay = np.exp(-50.0 / 100.0)
        self.assertTrue(
            torch.allclose(mag_flat, torch.tensor(expected_decay).float(), atol=1e-5)
        )

        # 2. Inhomogeneous field -> Signal loss (T2*)
        # Create a linear gradient B0 map
        # 100 Hz per pixel -> Strong dephasing
        # B0 [Hz] = gamma * B0_T
        # We pass B0 map in Tesla.
        # 100 Hz -> 100 / (42.58e6) Tesla
        gamma_hz = 42.58e6
        grad_amplitude_hz = 500.0
        # Create a map that varies 0 to 500 Hz across the voxel?
        # The solver computes gradients of the map.

        # Create a map with large gradient across pixels
        # [1, 10, 10]
        # x-gradient
        x = torch.linspace(0, 1, 10)
        b0_map_grad = x.unsqueeze(0).unsqueeze(0).expand(1, 10, 10) * (
            grad_amplitude_hz / gamma_hz
        )

        signal_dephased = solver.forward(m0, b0_map=b0_map_grad)
        mag_dephased = torch.norm(signal_dephased, dim=-1)

        # Signal should be significantly lower than T2 decay alone
        # But wait, Monte Carlo samples sub-voxel.
        # Finite difference gradient is computed from the map.
        # Map step is (grad_amplitude_hz / gamma_hz) / voxel_size? NO.
        # Map values are at pixel centers.
        # Gradient is d(B0)/dx.
        # Map changes by X per pixel. Solver calculates gradient from neighbors.
        # So Intra-voxel gradient is estimated from Inter-voxel changes.

        print(
            f"Flat Signal: {mag_flat.mean().item():.3f}, Dephased: {mag_dephased.mean().item():.3f}"
        )
        self.assertLess(
            mag_dephased.mean().item(),
            mag_flat.mean().item() * 0.95,
            "Inhomogeneous field should cause dephasing",
        )

    def test_stejskal_tanner_diffusion(self):
        """Verify exactness of Stejskal-Tanner diffusion attenuation."""
        # D = 1e-3 mm^2/s (Water approx is 3e-3 at 37C, but let's use 1.0)
        D = 1.0
        solver = BlochEquationSolver(diffusion_coeff=D, te=100.0)  # ms? te is sec.
        # Solver expects te in SECONDS.
        # D in mm^2/s.

        # TE = 0.1 s
        te = 0.1

        # Gradient G = 10 mT/m = 0.01 T/m
        G = 0.01
        gradient = torch.tensor([G, 0.0, 0.0])

        # Analytical b-value
        # Defaults: delta = te/3, Delta = te/2
        delta = te / 3.0
        Delta = te / 2.0
        gamma_hz = 42.58e6
        gamma_rad = 2 * np.pi * gamma_hz

        # b = [gamma * G * delta]^2 * (Delta - delta/3)
        # Note: In solver code I used gamma_rad.
        # Solver: b_val matches this formula.

        # But let's verify the calculated value matches external calc
        b_val_si = (gamma_rad * G * delta) ** 2 * (Delta - delta / 3.0)  # s/m^2
        b_val_mm = b_val_si * 1e-6  # s/mm^2

        expected_att = np.exp(-b_val_mm * D)

        # Run solver helper directly
        att = solver.compute_diffusion_attenuation(gradient, te)

        print(f"Diffusion Attenuation: {att.item():.4f}, Expected: {expected_att:.4f}")
        self.assertTrue(
            torch.allclose(att, torch.tensor(expected_att).float(), rtol=1e-3)
        )

    def test_maxwell_concomitant_field(self):
        """Verify Maxwell terms exist and scale correctly."""
        sim = B0MapSimulator(field_strength=0.064)  # 64 mT

        # 1. Zero gradient -> Zero Maxwell
        shape = (10, 10)
        field = sim.generate_concomitant_field(
            shape, field_strength=0.064, gx=0.0, gy=0.0, gz=0.0
        )
        self.assertEqual(field.abs().sum().item(), 0.0)

        # 2. Non-zero gradient -> Non-zero field
        # Note: generated field depends on r^2 (FOV).
        # Assuming Gx application at a slight z-offset to see effects
        field = sim.generate_concomitant_field(
            shape, field_strength=0.064, gx=10.0, z_offset=0.01
        )
        self.assertGreater(field.abs().sum().item(), 0.0)

        # 3. Scale check: 2x Gradient -> 4x Field
        field_2x = sim.generate_concomitant_field(
            shape, field_strength=0.064, gx=20.0, z_offset=0.01
        )
        ratio = field_2x.abs().mean() / field.abs().mean()
        self.assertTrue(torch.allclose(ratio, torch.tensor(4.0), rtol=0.1))

        # 4. Transverse term (z-offset)
        # At isocenter (z=0), only longitudinal term exists (r^2 dependence)
        # At z != 0, transverse term adds constant offset (Gx^2 z^2 / 2B0)
        field_off = sim.generate_concomitant_field(
            shape, field_strength=0.064, gx=10.0, z_offset=0.1
        )
        self.assertGreater(field_off.abs().mean(), field.abs().mean())

    def test_gaussian_fourier_gradients(self):
        """Verify analytical gradients using gradcheck."""
        # Setup small problem
        N = 3  # Gaussians
        M = 5  # K-space points
        means = torch.randn(N, 3, dtype=torch.float64, requires_grad=True)
        scales = torch.rand(N, 3, dtype=torch.float64, requires_grad=True)
        quats = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
        features = torch.randn(
            N, 1, dtype=torch.float64, requires_grad=True
        )  # Real features for testing
        k_traj = torch.randn(M, 3, dtype=torch.float64, requires_grad=False)

        # Wrapper for gradcheck inputs
        def func(m, s, q, f):
            # features need to be complex usually, but let's test real->complex flow
            # The op returns complex. Gradcheck handles complex outputs if valid?
            # PyTorch gradcheck supports complex.
            f_c = torch.complex(f, torch.zeros_like(f))
            return differentiable_gaussian_fourier(m, s, q, f_c, k_traj)

        # Grad check
        # We need double precision ensuring
        # inputs are float64.

        # Note: differentiable_gaussian_fourier casts features to match transform (complex).
        # We pass real features to 'func' which makes them complex.

        # gradcheck might fail if N/M are too small/condition number issues, but let's try.
        # Also need to handle optional args? No, passed all.

        test_passed = torch.autograd.gradcheck(
            func, (means, scales, quats, features), eps=1e-6, atol=1e-4
        )
        self.assertTrue(test_passed)


if __name__ == "__main__":
    unittest.main()
