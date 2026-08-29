import pytest
import torch

from mriforge.models.reconstruction.unet import (
    EnhancedUNetGenerator,
    UNetConfig,
    create_enhanced_unet_generator,
)


def test_enhanced_unet_device_handling(minimal_device):
    """Test that EnhancedUNetGenerator handles devices correctly.

    This ensures that when the model is moved to a device (GPU/CPU),
    and inputs are on that device, the forward pass works and output
    is on the correct device.
    """
    batch_size = 1
    channels = 1
    size = 32

    # Create model and move to correct device
    model = create_enhanced_unet_generator(
        in_channels=channels, out_channels=channels, use_attention=True
    ).to(minimal_device)

    # Create input on SAME device
    x = torch.randn(batch_size, channels, size, size, device=minimal_device)

    # Forward pass
    try:
        y = model(x)
    except RuntimeError as e:
        pytest.fail(f"Forward pass failed with runtime error: {e}")

    # Assert output is on same device
    assert (
        y.device.type == minimal_device.type
    ), f"Expected output on {minimal_device}, got {y.device}"
    assert y.shape == (batch_size, channels, size, size)


def test_unet_config_initialization(minimal_device):
    """Test UNet instantiation via UNetConfig."""
    config = UNetConfig(
        in_channels=1, out_channels=1, features=(16, 32), depth=2, use_attention=False
    )
    model = EnhancedUNetGenerator(config).to(minimal_device)
    x = torch.randn(1, 1, 32, 32, device=minimal_device)
    y = model(x)

    assert y.device.type == minimal_device.type
    assert y.shape == x.shape
