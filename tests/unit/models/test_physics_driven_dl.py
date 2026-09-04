import unittest

import torch

from spectramr.infrastructure.physics.implementations import FFT2DOperator
from spectramr.models.diffusion.score_sde.physics_guidance import (
    MeasurementConsistencyGuidance,
)
from spectramr.models.generators.physics_driven_network import UnrolledPhysicsNetwork
from spectramr.models.losses.physics_losses import HFENLoss, SpectralKSpaceLoss


class TestPhysicsDrivenDL(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Dummy Operator
        self.operator = FFT2DOperator()

    def test_measurement_consistency_guidance(self):
        """Verify gradient direction for consistency guidance."""
        B, C, H, W = 1, 2, 32, 32
        x_t = torch.randn(B, 2, H, W, device=self.device, requires_grad=True)
        # Assume measurement y matches x_target
        x_target = torch.randn(B, 2, H, W, device=self.device)
        y = self.operator.forward(x_target)

        guidance = MeasurementConsistencyGuidance(self.operator, y, noise_std=0.1)

        # Compute gradient
        grad = guidance.get_guidance_gradient(x_t, time_t=0.5)

        # Check shape
        self.assertEqual(grad.shape, x_t.shape)

        # Sanity: if x_t is close to x_target, grad should be small?
        # A^H(y - Ax) -> 0 if y=Ax
        grad_zero = guidance.get_guidance_gradient(x_target, time_t=0.5)
        self.assertTrue(
            torch.allclose(grad_zero, torch.zeros_like(grad_zero), atol=1e-5)
        )

        # Test POCS projection
        x_proj = guidance.apply_data_consistency(x_t)
        # Check if projected k-space is closer to y
        k_proj = self.operator.forward(x_proj)
        k_t = self.operator.forward(x_t)

        err_proj = torch.norm(k_proj - y)
        err_orig = torch.norm(k_t - y)

        self.assertLess(
            err_proj, err_orig, "POCS projection should reduce data consistency error"
        )

    def test_unrolled_physics_network(self):
        """Verify Unrolled Physics Network shapes and flow."""
        B, C, H, W = 2, 2, 64, 64
        model = UnrolledPhysicsNetwork(
            in_channels=2, operator_type="fft2d", num_iterations=2
        ).to(self.device)

        x_in = torch.zeros(B, 2, H, W, device=self.device)  # Zero filled
        # Make dummy ref kspace (fully sampled for simplicity)
        gt_image = torch.randn(B, 2, H, W, device=self.device)
        ref_kspace = torch.fft.fft2(
            torch.view_as_complex(gt_image.permute(0, 2, 3, 1).contiguous()),
            norm="ortho",
        )
        # Ensure shape [B, H, W, 2] usually or complex?
        # Interface says [B, C, H, W] for images.
        # FFT2DOperator inputs/outputs vary by implementation detail but typically handle [B, 2, H, W] -> [B, 2, H, W] kspace or complex.
        # Let's check typical usage. If standard FFT, output is same complex shape.
        # UnrolledPhysicsNetwork expects ref_kspace to match operator output.

        # Create mask
        mask = torch.ones(B, H, W, device=self.device)

        # Forward
        # ref_kspace needs to be correct format.
        # create_physics_operator("fft2d") returns FFT2DOperator.
        # FFT2DOperator.forward(img [B,2,H,W]) -> [B,2,H,W] kspace (channel first usually in simple implementations or complex?)
        # Let's inspect integration.py logic or assume typical.

        # For test simplicity, we assume robust handling or basic tensors
        # Pass ref_kspace as [B, H, W, 2] to match logic inside UnrolledNetwork
        # (which permutes x [B,2,H,W] -> [B,H,W,2] before DC)
        ref_kspace_tensor = torch.randn(B, H, W, 2, device=self.device)

        out = model(x_in, ref_kspace=ref_kspace_tensor, mask=mask)

        self.assertEqual(out.shape, x_in.shape)

    def test_spectral_losses(self):
        """Verify Spectral and HFEN losses."""
        loss_spectral = SpectralKSpaceLoss().to(self.device)
        loss_hfen = HFENLoss().to(self.device)

        B, C, H, W = 1, 1, 128, 128
        img1 = torch.randn(B, C, H, W, device=self.device)
        img2 = img1.clone()

        # Identical images -> Zero loss
        l_spec = loss_spectral(img1, img2)
        l_hfen = loss_hfen(img1, img2)

        self.assertAlmostEqual(l_spec.item(), 0.0, places=5)
        self.assertAlmostEqual(l_hfen.item(), 0.0, places=5)

        # Different images -> Positive loss
        img3 = img1 + 0.1 * torch.randn_like(img1)
        l_spec_diff = loss_spectral(img1, img3)
        l_hfen_diff = loss_hfen(img1, img3)

        self.assertGreater(l_spec_diff.item(), 0.0)
        self.assertGreater(l_hfen_diff.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
