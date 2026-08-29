import torch

from mriforge.models.generators.bloch_cycle_network import BlochCycleNetwork


def test_bloch_cycle_forward():
    # B, C, H, W
    x = torch.randn(2, 1, 64, 64)
    model = BlochCycleNetwork(in_channels=1, features=(32, 64))

    # Forward
    out = model(x)
    assert out.shape == (2, 1, 64, 64)

    # Check maps intermediate
    out, intermediates = model.forward_with_intermediates(x)
    maps = intermediates[0]
    # Maps should be (B, 3, H, W) -> PD, T1, T2
    assert maps.shape == (2, 3, 64, 64)
    # T1/T2 should be positive
    assert (maps >= 0).all()


def test_bloch_cycle_gradients():
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    model = BlochCycleNetwork(depth=2, features=(16, 32))
    out = model(x)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
