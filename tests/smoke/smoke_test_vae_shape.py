import torch

from spectramr.models.generators.vae_3d_generator import VAE3DGenerator


def test_vae_shape_compatibility():
    """Verify that VAE3DGenerator initializes and forwards correctly with non-cubic input."""
    input_shape = (64, 64, 32)
    in_channels = 1

    print(f"Testing VAE3DGenerator with input_shape={input_shape}...")

    try:
        model = VAE3DGenerator(
            in_channels=in_channels,
            latent_dim=512,
            hidden_dims=[32, 64, 128, 256],
            input_shape=input_shape,
        )
        print("Model initialized successfully.")

        # Verify feature size calculation
        expected_bottleneck = (
            input_shape[0] // 16,
            input_shape[1] // 16,
            input_shape[2] // 16,
        )
        print(
            f"Input: {input_shape} -> Divisor 16 -> Expected Bottleneck: {expected_bottleneck}"
        )
        print(f"Encoder Feature Size: {model.encoder.feature_size}")

        # Create dummy input
        x = torch.randn(2, in_channels, *input_shape)

        # Forward pass
        print("Running forward pass...")
        recon, mu, log_var = model(x)

        print(f"Output shape: {recon.shape}")

        assert (
            recon.shape == x.shape
        ), f"Output shape {recon.shape} mistmatch with input {x.shape}"
        print("SUCCESS: Forward pass complete and shapes match.")

    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    test_vae_shape_compatibility()
