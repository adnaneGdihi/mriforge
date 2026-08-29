import torch

from mriforge.models.mae.simae_generator import SiMAEGenerator


def test_simae_initialization():
    model = SiMAEGenerator(
        img_size=(32, 32, 32),
        patch_size=(8, 8, 8),
        embed_dim=128,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=8,
    )
    assert model is not None


def test_simae_forward():
    model = SiMAEGenerator(
        img_size=(32, 32, 32),
        patch_size=(8, 8, 8),
        embed_dim=128,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=8,
    )
    x = torch.randn(1, 1, 32, 32, 32)

    # Self-recon
    out = model(x)
    assert out.shape == x.shape

    # Style transfer
    style_ref = torch.randn(1, 1, 32, 32, 32)
    out_style = model(x, style_reference=style_ref)
    assert out_style.shape == x.shape
