"""Hermitian-equivariant FNO models (Idea 1 of audit_plan_novel).

Registers two architectures:

- ``hermitian_fno`` — pure FNO with Hermitian-equivariant spectral
  blocks throughout. Drop-in replacement for the existing ``fno``
  model.
- ``hermitian_kspace_unet`` — UNet whose bottleneck is replaced with
  a stack of Hermitian-FNO blocks. Useful when the input/output is
  real-image but the architecture should still respect the Hermitian
  invariant in the latent k-space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.hermitian_equivariant import HermitianFNOBlock
from mriforge.models.registry import register_model


@register_model(
    "hermitian_fno",
    training_mode="reconstruction",
    spatial_dims=(2,),
    # The FNO does its Fourier transform INTERNALLY on real feature maps
    # (in/out are real image-space tensors; the "Hermitian" refers to the
    # spectral blocks' equivariance, not a k-space input) — image-domain.
    # Accepts both 1-ch magnitude image and 2-ch real/imag (complex_image) —
    # the FNO lifts whatever real channels it is given (probe-verified on 2ch).
    input_domain=("image", "complex_image"),
    output_domain=("image", "complex_image"),
)
class HermitianFNO(nn.Module):
    """Pure FNO with Hermitian-equivariant spectral blocks.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        width: Hidden width.
        modes: Spectral modes (square block, ``modes × modes``).
        depth: Number of stacked Hermitian-FNO blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([HermitianFNOBlock(width, modes, modes) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.project(h)


@register_model(
    "hermitian_kspace_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    # Despite the name, the docstring is explicit: "Final output is real-valued
    # in image space"; in/out are real image-space tensors (the Hermitian-FNO
    # is only the bottleneck). Image-domain.
    # Accepts both 1-ch magnitude image and 2-ch real/imag (complex_image) —
    # the FNO lifts whatever real channels it is given (probe-verified on 2ch).
    input_domain=("image", "complex_image"),
    output_domain=("image", "complex_image"),
)
class HermitianKSpaceUNet(nn.Module):
    """U-Net with a Hermitian-FNO bottleneck.

    The encoder and decoder are vanilla strided convolutions; the
    bottleneck uses ``depth`` Hermitian-FNO blocks at the lowest
    resolution. Final output is real-valued in image space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 32,
        modes: int = 12,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(base_width, 2 * base_width, 2, stride=2)
        self.bottleneck = nn.ModuleList(
            [HermitianFNOBlock(2 * base_width, modes, modes) for _ in range(depth)]
        )
        self.up = nn.ConvTranspose2d(2 * base_width, base_width, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * base_width, base_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        e1 = self.enc1(x)
        b = self.down(e1)
        for block in self.bottleneck:
            b = block(b)
        u = self.up(b)
        return self.dec1(torch.cat([u, e1], dim=1))


__all__ = ["HermitianFNO", "HermitianKSpaceUNet"]
