import torch

from mriforge.models.blocks.block_registry import create_block


def test_unet_blocks_dispatch_and_shapes():
    x = torch.randn(1, 8, 16, 16)
    down = create_block("unet_down", in_channels=8, out_channels=8)
    x_down = down(x)
    assert x_down.shape[1] == 8

    up = create_block("unet_up", in_channels=8, out_channels=8, bilinear=True)
    x_up = up(x_down, x)
    assert x_up.shape[1] == 8

    out_conv = create_block("unet_out_conv", in_channels=8, out_channels=1)
    y = out_conv(x_up)
    assert y.shape[1] == 1


def test_attention_blocks_dispatch():
    x = torch.randn(1, 8, 16, 16)

    self_attn = create_block("self_attention", channels=8, num_heads=2)
    y = self_attn(x)
    assert y.shape == x.shape

    cross_attn = create_block("cross_attention", channels=8, num_heads=2)
    ctx = torch.randn(1, 8, 16, 16)
    y2 = cross_attn(x, context=ctx)
    assert y2.shape == x.shape

    spatial = create_block("spatial_attention", in_channels=8)
    y3 = spatial(x)
    assert y3.shape == x.shape
