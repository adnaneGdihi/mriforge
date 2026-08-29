import torch

from mriforge.models.components.conditioning_encoder import ConditioningEncoder


class TestConditioningEncoder:
    def test_output_shape(self):
        """Test that encoder produces correct downsampled latent shape."""
        # Setup
        B, C, H, W = 2, 3, 256, 256
        latent_channels = 4
        levels = 3  # 2^3 = 8x downsampling

        encoder = ConditioningEncoder(
            in_channels=C,
            out_channels=latent_channels,
            base_channels=32,
            num_levels=levels,
        )

        x = torch.randn(B, C, H, W)

        # Execute
        out = encoder(x)

        # Verify
        expected_shape = (B, latent_channels, H // (2**levels), W // (2**levels))
        assert (
            out.shape == expected_shape
        ), f"Expected {expected_shape}, got {out.shape}"

    def test_gradients(self):
        """Test that gradients flow through the encoder."""
        encoder = ConditioningEncoder(base_channels=16)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_channels_handling(self):
        """Test with different channel configurations."""
        encoder = ConditioningEncoder(in_channels=1, out_channels=8)
        x = torch.randn(1, 1, 64, 64)
        out = encoder(x)
        assert out.shape[1] == 8
