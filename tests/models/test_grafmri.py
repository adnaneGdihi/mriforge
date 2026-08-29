import torch

from mriforge.models.geometric.grafmri import GraFMRIGenerator


def test_grafmri_initialization():
    model = GraFMRIGenerator(
        in_channels=1, out_channels=1, patch_size=4, hidden_dim=32, k=5, layers=2
    )
    assert model.name == "GraFMRI"


def test_grafmri_forward():
    # Setup
    patch_size = 4
    H, W = 64, 64
    B = 2
    C = 1

    model = GraFMRIGenerator(
        in_channels=C,
        out_channels=C,
        patch_size=patch_size,
        hidden_dim=32,
        k=5,
        layers=2,
    )

    x = torch.randn(B, C, H, W)
    out = model(x)

    assert out.shape == (B, C, H, W)
    assert not torch.isnan(out).any()


def test_grafmri_features_dynamic():
    """Verify dynamic graph construction doesn't crash."""
    model = GraFMRIGenerator(k=4, layers=2)
    x = torch.randn(1, 1, 32, 32)
    out1 = model(x)

    # Check output range or shape
    assert out1.shape == (1, 1, 32, 32)
