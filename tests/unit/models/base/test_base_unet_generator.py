import pytest
import torch
from torch import nn

from spectramr.models.base.base_unet_generator import BaseUNetGenerator


class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels to conv is in_channels (from upsample) + out_channels (from skip)
        # Wait, usually in_channels is the deeper feature map.
        # If we concat, the channels increase.
        # Let's assume simple concat.
        self.conv = nn.Conv2d(
            in_channels + out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            # Center crop if needed, but for test we assume same size
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConcreteUNet(BaseUNetGenerator):
    def _make_encoder_block(self, in_channels, out_channels, include_pool=True):
        layers = []
        if include_pool:
            layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels, out_channels):
        return SimpleDecoderBlock(in_channels, out_channels)


class TestBaseUNetGenerator:
    def test_init(self):
        unet = ConcreteUNet(
            in_channels=1, out_channels=1, depth=2, features=(16, 32, 64)
        )
        assert unet.in_channels == 1
        assert unet.out_channels == 1
        assert unet.depth == 2
        assert len(unet.encoder) == 3  # depth + 1
        assert len(unet.decoder) == 2  # depth

    def test_init_invalid_features(self):
        with pytest.raises(ValueError, match="features tuple must have at least"):
            ConcreteUNet(depth=3, features=(16, 32))

    def test_forward(self):
        unet = ConcreteUNet(
            in_channels=1, out_channels=1, depth=2, features=(16, 32, 64)
        )
        x = torch.randn(1, 1, 32, 32)
        out = unet(x)
        assert out.shape == (1, 1, 32, 32)

    def test_generate(self):
        unet = ConcreteUNet(in_channels=1, out_channels=1, depth=1, features=(16, 32))
        x = torch.randn(1, 1, 16, 16)
        out = unet.generate(x)
        assert out.shape == (1, 1, 16, 16)

    def test_get_output_shape(self):
        unet = ConcreteUNet(in_channels=1, out_channels=3)
        shape = unet.get_output_shape((1, 1, 64, 64))
        assert shape == (1, 3, 64, 64)

    def test_get_parameter_count(self):
        unet = ConcreteUNet(in_channels=1, out_channels=1, depth=1, features=(16, 32))
        assert unet.get_parameter_count() > 0
