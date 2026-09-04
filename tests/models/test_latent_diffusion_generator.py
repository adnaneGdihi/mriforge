import pytest
import torch

from spectramr.models.generators.latent_diffusion_generator import (
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)


class TestLatentDiffusionGenerator:
    @pytest.fixture
    def generator(self):
        config = LatentDiffusionGeneratorConfig(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            base_channels=32,
            timesteps=10,  # Small number for speed
            spatial_dims=2,
            device="cpu",
        )
        # Use num_layers=3 to get downsample_factor=8 (2^3)
        model = LatentDiffusionGenerator(config, num_layers=3)
        return model

    def test_initialization(self, generator):
        assert isinstance(generator, LatentDiffusionGenerator)
        assert generator.diffusion.timesteps == 10

    def test_forward(self, generator):
        # Input shape (B, C, H, W)
        # Size 64 is divisible by default downsample factor 16 (4 layers)
        x = torch.randn(2, 4, 16, 16)  # latent shape
        t = torch.randint(0, 10, (2,))
        output = generator(x, timesteps=t)
        assert output.shape == (2, 4, 16, 16)
        assert not torch.isnan(output).any()

    def test_sample(self, generator):
        # Sample shape (B, C, H, W)
        output = generator.sample(shape=(2, 1, 64, 64))
        # Verify correct upsampling back to original resolution
        assert output.shape == (2, 1, 64, 64)
        assert not torch.isnan(output).any()

    def test_forward_3d(self):
        config = LatentDiffusionGeneratorConfig(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            base_channels=16,
            timesteps=5,
            spatial_dims=3,
            device="cpu",
        )
        generator = LatentDiffusionGenerator(config)

        # 3D input: (B, C, D, H, W)
        # Latent shape: (B, 4, D/8, H/8, W/8)
        x = torch.randn(1, 4, 2, 4, 4)
        t = torch.randint(0, 5, (1,))
        output = generator(x, timesteps=t)
        assert output.shape == (1, 4, 2, 4, 4)
        assert not torch.isnan(output).any()
