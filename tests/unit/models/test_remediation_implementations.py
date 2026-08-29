import torch
import torch.nn as nn

from mriforge.infrastructure.physics.dynamics.neural_ode import NeuralODEDynamics
from mriforge.infrastructure.physics.rectified_flow import RectifiedFlow
from mriforge.models.diffusion.cold_diffusion import ColdDiffusion
from mriforge.models.generators.kspace_cold_diffusion_generator import PureKSpaceUNet
from mriforge.models.generators.nesvor import NeSVoR
from mriforge.models.vae.wavelet_kan_vae import WaveletKANVAE


class TestRemediationImplementations:
    def test_rectified_flow(self):
        """Exp 48: Verify Rectified Flow Physics."""
        rf = RectifiedFlow()
        x0 = torch.randn(2, 64)
        x1 = torch.randn(2, 64)

        # Test 1: Velocity Target
        # v = x1 - x0
        v_target = rf.get_velocity_target(x0, x1)
        assert torch.allclose(v_target, x1 - x0), "Velocity target computation failed"

        # Test 2: Euler Step
        # x_next = x + v * dt
        dt = 0.1
        v_pred = torch.ones_like(x0)
        x_next = rf.step_euler(x0, v_pred, dt)
        assert torch.allclose(x_next, x0 + v_pred * dt), "Euler integration step failed"

    def test_neural_ode(self):
        """Exp 52: Verify Neural ODE Solver."""

        # Define a simple drift: dx/dt = -x => x(t) = x0 * exp(-t)
        class DecayDrift(nn.Module):
            def forward(self, x, t):
                return -x

        # Use simple Euler for deterministic check if library missing or checking logic
        # But NeuralODEDynamics might use dopri5.
        # For small interval, result should be close.
        ode = NeuralODEDynamics(drift_net=DecayDrift(), solver="euler")

        x0 = torch.ones(1, 1)  # Batch 1, dim 1
        t = torch.tensor([0.0, 1.0])

        # If torchdiffeq not installed, it falls back to manual euler (step 1 for t=0->1?)
        # Manual integration in code splits t into steps?
        # My manual integration implementation does: for i in range(len(t)-1).
        # So it takes one big step if t has 2 points.
        # Euler step 1.0: x1 = x0 + (-x0)*1 = 0.
        # Analytic: exp(-1) = 0.367.
        # To get closer, we need more t points.

        t_fine = torch.linspace(0, 1, 11)  # dt = 0.1
        res = ode(x0, t_fine)

        # Check shape: (B, T, D) -> (1, 11, 1)
        assert res.shape == (1, 11, 1), f"Shape Mismatch: {res.shape}"

        # Check last value with loose tolerance (Euler is approximation)
        # x_final approx x0 * (1 - 0.1)^10 = 0.9^10 = 0.348
        # Analytic = 0.367
        # Should be reasonably close.
        assert (
            torch.abs(res[0, -1, 0] - torch.exp(torch.tensor(-1.0))) < 0.1
        ), "ODE Integration diverged too much"

    def test_nesvor(self):
        """Exp 69: Verify NeSVoR INR Shape."""
        model = NeSVoR(hidden_dim=32)
        B, N = 2, 100
        coords = torch.randn(B, N, 3)
        time = torch.randn(B, 1)

        out = model(coords, time)

        # Expected: (B, N, 1) Intensity
        assert out.shape == (B, N, 1), f"NeSVoR Output shape failed: {out.shape}"

    def test_wavelet_kan_vae(self):
        """Exp 93: Verify Wavelet KAN VAE."""
        # Using 1 channel, 128 latent
        vae = WaveletKANVAE(in_channels=1, latent_dim=16)

        # Input: (B, 1, 64, 64) - Assuming logic supports 64x64
        # Encoder: 64 -> 32 -> 16 (2 max pools)
        # Flatten: 16*16 = 256. 64 channels. 64*256 = 16384 dims.
        # Code has `64*16*16` hardcoded.
        x = torch.randn(2, 1, 64, 64)

        recon, mu, logvar = vae(x)

        assert recon.shape == (2, 1, 64, 64), f"Recon shape mismatch: {recon.shape}"
        assert mu.shape == (2, 16), "Latent shape mismatch"

    def test_kspace_cold_diffusion_pure_unet(self):
        """Exp 11: Verify Pure K-Space U-Net."""
        # Test the Helper Class directly
        model = PureKSpaceUNet(in_channels=2, out_channels=2, features=(16, 32))

        # Input (B, 2, 64, 64)
        x = torch.randn(2, 2, 64, 64)
        out = model(x)

        # Expect same shape
        assert out.shape == x.shape, "PureKSpaceUNet output shape mismatch"
        assert not torch.isnan(out).any(), "NaN in PureKSpaceUNet output"

    def test_cold_diffusion_data_consistency(self):
        """Exp 11: Verify Data Consistency in Sampling."""

        # Mock Model: Identity
        class MockModel(nn.Module):
            def forward(self, x, t, cond=None):
                return x

        model = MockModel()

        # Initialize ColdDiffusion with hard DC
        cd = ColdDiffusion(
            timesteps=1, model=model, dc_method="hard", degradation_type="kspace_mask"
        )

        # Measured K-space (Ground Truth Anchor). Use a random k-space tensor
        # so that we can verify hard DC anchors the *image-domain* output to
        # ifft2c(measured) — i.e. data consistency is exact in k-space.
        torch.manual_seed(0)
        measured = torch.randn(1, 2, 32, 32)

        # Mask (all 1s means we have all measurements) — per ProximalDCStep
        # contract: (B, 1, H, W) or broadcastable, not duplicated along channels.
        mask = torch.ones(1, 1, 32, 32)

        # Run 1 step of sampling
        out = cd.generate(input_data=measured, mask=mask)

        # Shape must be preserved.
        assert out.shape == measured.shape, (
            f"Expected output shape {measured.shape}, got {out.shape}"
        )

        # With mask=1 everywhere, the hard DC projection in k-space replaces
        # every frequency with the measurement. The expected image-domain
        # output is therefore ifft2c(measured).
        from mriforge.infrastructure.physics.fft_ops import _to_complex, ifft2c

        expected_complex = ifft2c(_to_complex(measured))
        expected = torch.cat([expected_complex.real, expected_complex.imag], dim=1)
        assert torch.allclose(out, expected, atol=1e-4), (
            "Data Consistency failed to anchor output to measurement"
        )
