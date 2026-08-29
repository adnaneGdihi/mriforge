import torch

from mriforge.models.diffusion.diffusion_parts.kan_diffusion import DiffusionKAN


class TestKANDiffusion:
    def test_complex_noise_injection_variance(self):
        """Verify that Q-sample generates noise with correct variance for 2-channel (complex) inputs."""
        diff = DiffusionKAN(timesteps=10)

        # B=1000, 2, 1, 1 (Statistical sample)
        x = torch.zeros(1000, 2, 1, 1)
        t = torch.zeros(1000, dtype=torch.long)  # t=0

        # q_sample at t=0
        # sqrt_alpha=1, sqrt_1_minus_alpha=0 (at t=0??)
        # Wait, betas_start=0.0001
        # alpha_0 = 0.9999
        # sqrt_one_minus_alpha = sqrt(0.0001) = 0.01

        # To test noise variance, let's use t where weight is substantial?
        # Or checking `noise` argument handling?

        # Let's inspect q_sample logic directly if we can, or just trust the scaling.
        # But statistical test is good.

        # x_t = sqrt_alpha * x + sqrt_1_minus_alpha * noise
        # if x=0. x_t is scaled noise.
        # variance(x_t) = (sqrt_1_minus_alpha)^2 * variance(noise)
        # variance(noise) should be 1.0 per channel.

        x_t = diff.q_sample(x, t)

        # Check variance of Re and Imag channels
        var_re = x_t[:, 0].var()
        var_im = x_t[:, 1].var()

        # Expected scaling factor
        expected_scale = diff.sqrt_one_minus_alphas_cumprod[0]
        expected_var = expected_scale**2 * 1.0  # Var(noise)=1.0

        # Check if actual var matches expected var
        # Note: tolerance needed for 1000 samples
        assert torch.allclose(
            var_re, expected_var, rtol=0.2
        ), f"Re Var {var_re} != {expected_var}"
        assert torch.allclose(
            var_im, expected_var, rtol=0.2
        ), f"Im Var {var_im} != {expected_var}"

    def test_complex_handling(self):
        """Verify 2-channel input is treated as complex."""
        diff = DiffusionKAN(timesteps=10)
        x = torch.randn(2, 2, 16, 16)
        t = torch.tensor([5, 5])

        out = diff.q_sample(x, t)
        assert out.shape == x.shape
        assert not torch.is_complex(out)  # Should return 2-channel real

        # Complex input
        x_c = torch.randn(2, 1, 16, 16, dtype=torch.complex64)
        out_c = diff.q_sample(x_c, t)
        # Should return complex
        assert torch.is_complex(out_c)
        assert out_c.shape == x_c.shape
