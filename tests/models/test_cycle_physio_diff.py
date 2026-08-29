import torch

from mriforge.models.generators.cycle_physio_diff import CyclePhysioDiffGenerator


def test_cycle_physio_diff_initialization():
    model = CyclePhysioDiffGenerator(
        in_channels=1,
        out_channels=1,
        img_size=64,
        diffusion_timesteps=10,
        window_size=4,
        conditional_translation=True,
    )
    assert model.name == "CyclePhysioDiff"


def test_cycle_physio_diff_forward_training():
    """Test forward pass (training mode)."""
    model = CyclePhysioDiffGenerator(
        in_channels=1,
        out_channels=1,
        img_size=64,
        diffusion_timesteps=10,
        embed_dim=24,  # Small for test
        window_size=4,
        conditional_translation=True,
    )
    model.train()

    B, C, H, W = 2, 1, 64, 64
    x = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)

    refined, coarse, q_maps = model(x, target=target)

    assert coarse.shape == (B, C, H, W)
    assert q_maps.shape == (B, 3, H, W)
    # refined should be diffusion output (noise prediction in latent space)
    # img_size 64 with 3 downsamples (f=8) -> 8x8 latent
    assert refined.shape == (B, 4, 8, 8)


def test_cycle_physio_diff_inference():
    """Test inference generation."""
    model = CyclePhysioDiffGenerator(
        in_channels=1,
        out_channels=1,
        img_size=32,
        diffusion_timesteps=5,
        embed_dim=24,
        window_size=4,  # 32/4 = 8, divisible by 4
        conditional_translation=True,
    )
    model.eval()

    B, C, H, W = 1, 1, 32, 32
    x = torch.randn(B, C, H, W)

    # Inference sampling
    with torch.no_grad():
        refined = model.generate(x)

    # Should upsample back to original resolution
    assert refined.shape == (B, C, H, W)
