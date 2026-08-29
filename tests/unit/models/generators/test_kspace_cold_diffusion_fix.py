import pytest
import torch

from mriforge.models.generators.kspace_cold_diffusion_generator import FourierBridgeNetwork
from mriforge.models.reconstruction.unet import UNetConfig


def test_fourier_bridge_network_complex_input_handling():
    """Test FourierBridgeNetwork with multi-channel inputs (simulating multi-coil MRI)."""

    # 1. Test standard 2-channel input (Real/Imag) — config matches input layout.
    config_2ch = UNetConfig(
        in_channels=2,
        out_channels=2,
        features=(32, 64),
        depth=2,
    )
    model = FourierBridgeNetwork(config_2ch)
    x_standard = torch.randn(1, 2, 32, 32)
    try:
        out = model(x_standard)
        assert out.shape == (1, 2, 32, 32)
        print("Standard 2-channel input passed.")
    except Exception as e:
        pytest.fail(f"Standard 2-channel input failed: {e}")

    # 2. Test multi-channel input (e.g. 2 coils -> 4 channels) - The bug scenario.
    # Config sets in_channels=4 and out_channels=4 to match the multi-coil layout.
    config_4ch = UNetConfig(
        in_channels=4,  # 2 coils = 4 channels (real/imag stacked)
        out_channels=4,
        features=(32, 64),
        depth=2,
    )
    model2 = FourierBridgeNetwork(config_4ch)
    x_multicoil = torch.randn(1, 4, 32, 32)
    try:
        out = model2(x_multicoil)
        assert out.shape == (1, 4, 32, 32)
        print("Multi-channel input passed.")
    except RuntimeError as e:
        if "Tensor must have a last dimension of size 2" in str(e):
            pytest.fail(
                "Caught the regression: Tensor must have a last dimension of size 2"
            )
        else:
            pytest.fail(f"RuntimeError: {e}")
    except Exception as e:
        pytest.fail(f"Multi-channel input failed with unexpected error: {e}")


if __name__ == "__main__":
    test_fourier_bridge_network_complex_input_handling()
