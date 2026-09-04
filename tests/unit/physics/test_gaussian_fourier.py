import math
import unittest

import torch
from torch.autograd import gradcheck


class TestGaussianFourierOp(unittest.TestCase):
    def test_gradient_check(self):
        """Task 1.6: Verify analytical gradients match finite differences."""
        device = "cpu"
        dtype = torch.float64  # Use double precision for gradcheck

        # Small setup
        N = 3  # Gaussians
        M = 5  # K-space points
        C = 1  # Channels (Complex)

        means = torch.randn(N, 3, device=device, dtype=dtype, requires_grad=True)
        scales = torch.rand(N, 3, device=device, dtype=dtype, requires_grad=True)
        quaternions = torch.randn(
            N, 4, device=device, dtype=dtype, requires_grad=True
        )  # Unused in diagonal mode but checked
        # Normalize quats for inputs if needed, but the Op currently ignores them or handles raw.
        # Op ignores quats in current implementation.

        features = torch.randn(
            N, C, device=device, dtype=dtype, requires_grad=True
        )  # Real components
        # Op expects complex feature handling?
        # The Op comments: "features: (N, C) - Complex weights (assumed complex64...)"
        # But implementation uses: `k_data = torch.sum(weights * features.unsqueeze(0), dim=1)`
        # If `features` is realfloat, result is complex (weights are complex).
        # gradcheck usually likes real inputs/outputs.

        k_trajectory = torch.randn(M, 3, device=device, dtype=dtype, requires_grad=True)

        # We need to wrap the autograd function to output a real scalar for gradcheck
        # or check complex gradients support.
        # torch.autograd.gradcheck supports complex now? Yes.

        # However, for simplicity, let's treat features as real magnitude for testing,
        # and checking grad of real/imag parts of output.

        # GaussianFourierOp.apply(means, scales, quaternions, features, k_trajectory)

        # Use a lambda that returns summation of absolute value or real part?
        # gradcheck checks partial derivatives.

        inputs = (means, scales, quaternions, features, k_trajectory)

        # GaussianFourierOp is a class, but we need a function for gradcheck if it's not a custom autograd Function.
        # The implementation differentiable_gaussian_fourier is a regular PyTorch function.
        # So we should call gradcheck on that function.
        # GaussianFourierOp.apply likely refers to torch.autograd.Function.apply which this is NOT.

        from spectramr.infrastructure.physics.implementations.gaussian_fourier_op import (
            differentiable_gaussian_fourier,
        )

        # features need to be cast to complex for logic inside if needed, but gradcheck likes matching types.
        # The implementation handles it.

        test = gradcheck(differentiable_gaussian_fourier, inputs, eps=1e-6, atol=1e-4)
        self.assertTrue(test, "Gradient check failed")

    def test_quaternion_orientation_is_wired(self):
        """Regression: `quaternions` must affect an anisotropic Gaussian's FT.

        Previously the orientation was ignored (axis-aligned decay only), so any
        rotation was a silent no-op (pitfall #15).
        """
        from spectramr.infrastructure.physics.implementations.gaussian_fourier_op import (
            differentiable_gaussian_fourier,
        )

        torch.manual_seed(0)
        means = torch.zeros(1, 3)
        scales = torch.tensor([[2.0, 0.5, 0.5]])  # anisotropic
        features = torch.ones(1, 1)
        k_traj = torch.randn(8, 3)
        q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity (wxyz)
        # 90 deg about z: (cos45, 0, 0, sin45)
        c = math.cos(math.pi / 4)
        q_z90 = torch.tensor([[c, 0.0, 0.0, c]])

        out_id = differentiable_gaussian_fourier(means, scales, q_id, features, k_traj)
        out_rot = differentiable_gaussian_fourier(means, scales, q_z90, features, k_traj)
        self.assertFalse(torch.allclose(out_id, out_rot, atol=1e-4))

        # Correctness: a 90 deg z-rotation of scales (sx, sy, sz) equals identity
        # with the x/y scales swapped (the covariance swaps its xx/yy entries).
        out_swap = differentiable_gaussian_fourier(
            means, torch.tensor([[0.5, 2.0, 0.5]]), q_id, features, k_traj
        )
        self.assertTrue(torch.allclose(out_rot, out_swap, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
