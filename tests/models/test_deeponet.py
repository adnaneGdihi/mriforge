import torch

from spectramr.models.generators.deeponet_generator import DeepONetGenerator


def test_deeponet_initialization():
    model = DeepONetGenerator(
        in_channels=2, basis_dim=32, out_channels=2, hidden_dim=32
    )
    assert model.name == "DeepONet"


def test_deeponet_forward():
    model = DeepONetGenerator(
        in_channels=2, basis_dim=16, out_channels=2, hidden_dim=32, img_size=(32, 32)
    )
    x = torch.randn(2, 2, 32, 32)
    out = model(x)

    assert out.shape == (2, 2, 32, 32)
    assert not torch.isnan(out).any()


def test_deeponet_custom_output_dims():
    """Verify implicit dense output concept for now."""
    # Our impl currently uses cached grid matching img_size unless we change get_grid
    # But forward uses get_grid(H, W) from input shape

    model = DeepONetGenerator(basis_dim=16)
    # Test variable resolution input
    x = torch.randn(1, 2, 64, 64)
    out = model(x)
    assert out.shape == (1, 2, 64, 64)
