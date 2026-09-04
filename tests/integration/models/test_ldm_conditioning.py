import torch

from spectramr.models.generators.latent_diffusion_generator import (
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)


class TestLDMConditioning:
    def test_forward_with_conditioning(self):
        """Test LDM forward pass with ULF conditioning image."""
        # Setup
        B, C_in, H, W = 2, 3, 64, 64  # Small size for speed
        latent_ch = 4

        config = LatentDiffusionGeneratorConfig(
            in_channels=C_in,
            out_channels=1,
            latent_channels=latent_ch,
            base_channels=16,
            conditional_translation=True,
            spatial_dims=2,
        )

        model = LatentDiffusionGenerator(config=config)

        # Mock inputs
        # VAE/Diffusion expects normalized inputs
        x_target = torch.randn(B, 1, H, W)  # HF Target
        x_cond = torch.randn(B, C_in, H, W)  # ULF Source
        timesteps = torch.randint(0, 1000, (B,))

        # Execute
        # LatentDiffusionGenerator.forward expects:
        # x: Noisy latent (B, C_latent, H_latent, W_latent)
        # timesteps
        # condition_image

        # Wait, forward signature I updated requires (x, timesteps, context, condition_image)
        # But usually forward in training loop takes images and handles everything?
        # Let's check LatentDiffusionGenerator.forward docstring I wrote/saw.
        # "Forward pass for training (noise prediction)."
        # "x: Noisy latent input (B, C, H, W)"

        # So I need to encode first or pass latents?
        # Usually training loop handles encoding. This test assumes input is already latent.

        latent_h, latent_w = H // 8, W // 8
        z_noisy = torch.randn(B, latent_ch, latent_h, latent_w)

        out = model(z_noisy, timesteps, condition_image=x_cond)

        # Verify
        assert out.shape == z_noisy.shape, f"Expected {z_noisy.shape}, got {out.shape}"

    def test_cfg_dropout_shapes(self):
        """Test that CFG dropout (via masking) maintains shapes."""
        B, C_in, H, W = 2, 3, 64, 64
        latent_ch = 4

        config = LatentDiffusionGeneratorConfig(
            in_channels=C_in,
            out_channels=1,
            latent_channels=latent_ch,
            base_channels=16,
            conditional_translation=True,
        )
        model = LatentDiffusionGenerator(config=config)
        model.train()  # Enable dropout logic

        latent_h, latent_w = H // 8, W // 8
        z_noisy = torch.randn(B, latent_ch, latent_h, latent_w)
        x_cond = torch.randn(B, C_in, H, W)
        timesteps = torch.randint(0, 1000, (B,))

        out = model(z_noisy, timesteps, condition_image=x_cond)
        assert out.shape == z_noisy.shape
