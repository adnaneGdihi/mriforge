"""Shape-robustness guard for LatentUNet skip-connection alignment (2026-07).

The decoder floors odd sizes on downsample but Upsample doubles exactly, so a
latent whose in-plane size is not a multiple of 2**(len(channel_mult)-1) used to
crash at ``torch.cat([h, skip])`` ("Expected size 10 but got size 9"). The fix
interpolates ``h`` to the skip's spatial size before concatenation, so ANY input
size is accepted and the output keeps the input geometry.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.latent_diffusion.latent_unet import LatentUNet


def _net():
    # channel_mult=(1,2,4) → 2 downsamples, the config the LDM actually builds.
    return LatentUNet(
        in_channels=4,
        out_channels=4,
        time_emb_dim=128,
        cond_channels=0,
        base_channels=16,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_levels=(),
        dropout=0.0,
    )


@pytest.mark.parametrize("size", [16, 18, 9, 33])
def test_forward_accepts_non_divisible_latent_sizes(size: int) -> None:
    net = _net().eval()
    x = torch.randn(1, 4, size, size)
    t = torch.randint(0, 5, (1,))
    out = net(x, t)
    # Output must keep the input geometry (the skip-align fix, not a crash).
    assert tuple(out.shape) == (1, 4, size, size)


def test_odd_size_would_have_crashed_without_fix() -> None:
    """18→9→5 (floor) then 5→10 upsample vs skip 9 was the exact 10-vs-9 crash."""
    net = _net().eval()
    out = net(torch.randn(1, 4, 18, 18), torch.randint(0, 5, (1,)))
    assert tuple(out.shape[-2:]) == (18, 18)
