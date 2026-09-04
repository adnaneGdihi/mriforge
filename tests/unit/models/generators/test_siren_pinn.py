import torch

from spectramr.models.generators.siren_pinn import (
    SirenLayer,
    SirenSensNet,
    get_last_shared_layer,
)


def test_siren_layer_initialization():
    in_features = 2
    out_features = 256
    layer_first = SirenLayer(in_features, out_features, is_first=True, omega_0=30.0)
    layer_hidden = SirenLayer(in_features, out_features, is_first=False, omega_0=30.0)

    assert layer_first.in_features == in_features
    assert layer_hidden.in_features == in_features

    # Check weight dimensions
    assert layer_first.linear.weight.shape == (out_features, in_features)
    assert layer_hidden.linear.weight.shape == (out_features, in_features)


def test_siren_layer_forward():
    layer = SirenLayer(2, 4, is_first=True, omega_0=30.0)
    x = torch.zeros(1, 2)
    out = layer(x)

    # weights are uniform, so x=0 gives bias. bias is 0 by default linear init?
    # actually nn.Linear init gives some bias. sin(0 + bias) will be something.
    assert out.shape == (1, 4)


def test_siren_sens_net_forward():
    net = SirenSensNet(
        in_features=2, hidden_features=64, hidden_layers=2, out_features=16
    )

    # Batch of 10 points
    coords = torch.rand(10, 2)
    s_real, s_imag = net(coords)

    assert s_real.shape == (10, 8)
    assert s_imag.shape == (10, 8)
    assert s_real.requires_grad is True


def test_get_last_shared_layer():
    net = SirenSensNet(
        in_features=2, hidden_features=64, hidden_layers=2, out_features=16
    )
    param = get_last_shared_layer(net)

    assert isinstance(param, torch.nn.Parameter)
    assert param.shape == (64, 64)
