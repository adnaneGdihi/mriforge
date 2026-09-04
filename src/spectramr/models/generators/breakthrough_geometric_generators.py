r"""Breakthrough 2026 geometric-prior generators.

Two proof-of-integration generators wrapping the geometric-prior blocks
from Batch 1 (Hyperbolic Poincaré, Heisenberg twisted-convolution) so
they are reachable from a YAML by ``model.model_type``. The pattern
mirrors :mod:`spectramr.models.generators.breakthrough_attention_generators`:
a small U-Net with the geometric block in the bottleneck.

* :class:`HyperbolicCineUNet` (P-Hyperbolic) — Poincaré-ball latent for
  hierarchical cardiac-cine reconstruction. The encoder lifts features
  into :math:`\mathbb{B}^d_c` via :class:`PoincareLayer`, applies a
  Möbius-add bottleneck, and projects back to Euclidean via
  :func:`log_map_zero`.
* :class:`HeisenbergReconUNet` (P-Heisenberg) — Heisenberg-equivariant
  k-space prior. The bottleneck uses one
  :class:`HeisenbergConv2d` layer; the output is complex-valued so we
  return its magnitude.

For the mathematical background see
``docs/breakthrough_methods.rst`` (Hyperbolic / Heisenberg sections).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.blocks.heisenberg_conv import HeisenbergConv2d
from spectramr.models.blocks.hyperbolic_layer import PoincareLayer, log_map_zero
from spectramr.models.registry import register_model

__all__ = ["HeisenbergReconUNet", "HyperbolicCineUNet"]


_BOTTLENECK_HW: int = 32
_BOTTLENECK_DIM: int = 16


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
        nn.SiLU(inplace=True),
    )


@register_model(
    name="hyperbolic_cine_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class HyperbolicCineUNet(nn.Module):
    r"""U-Net with a Poincaré-ball hyperbolic bottleneck (Batch-1 #5).

    The Poincaré bias is added via :func:`mobius_add` after the linear lift
    by :class:`PoincareLayer`; a :func:`log_map_zero` step projects back to
    Euclidean before the decoder.

    Args:
        in_channels: Input image channels.
        out_channels: Output image channels.
        curvature: Poincaré-ball curvature parameter :math:`c\ge 0`.
            Default 1.0; ``c=0`` recovers the Euclidean baseline.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        if curvature < 0:
            raise ValueError("curvature must be non-negative.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature = curvature
        self.encoder_1 = _conv_block(in_channels, 16)
        self.encoder_2 = _conv_block(16, _BOTTLENECK_DIM)
        self.poincare = PoincareLayer(
            in_features=_BOTTLENECK_DIM,
            out_features=_BOTTLENECK_DIM,
            curvature=curvature,
            bias=True,
        )
        self.decoder_2 = _conv_block(_BOTTLENECK_DIM, 16)
        self.decoder_1 = _conv_block(16 + 16, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        feat1 = self.encoder_1(x)
        pooled = self.down(feat1)
        feat2 = self.encoder_2(pooled)
        # Resize bottleneck to fixed token grid.
        feat2_resized = nn.functional.interpolate(
            feat2, size=(_BOTTLENECK_HW, _BOTTLENECK_HW), mode="bilinear", align_corners=False
        )
        b, c, h, w = feat2_resized.shape
        tokens = feat2_resized.reshape(b, c, h * w).transpose(1, 2).reshape(-1, c)
        hyp = self.poincare(tokens)
        # Möbius-add a learned bias (already absorbed inside PoincareLayer)
        # then project back via log_map_zero.
        eucl = log_map_zero(hyp, c=self.curvature)
        feat_back = eucl.reshape(b, h * w, c).transpose(1, 2).reshape(b, c, h, w)
        pooled_hw = (original_size[0] // 2, original_size[1] // 2)
        upsampled = nn.functional.interpolate(
            feat_back, size=pooled_hw, mode="bilinear", align_corners=False
        )
        dec2 = self.decoder_2(upsampled)
        dec2 = self.up(dec2)
        merged = torch.cat([dec2, feat1], dim=1)
        dec1 = self.decoder_1(merged)
        return self.head(dec1)


@register_model(
    name="heisenberg_recon_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
)
class HeisenbergReconUNet(nn.Module):
    r"""U-Net with a Heisenberg-equivariant twisted-convolution bottleneck (Batch-1 #6).

    The bottleneck applies one :class:`HeisenbergConv2d` layer whose
    output is complex; the decoder consumes the magnitude. Acts on small
    inputs by interpolating to a fixed bottleneck spatial size.

    Args:
        in_channels: Input image channels.
        out_channels: Output image channels.
        translation_radius: Heisenberg-lattice translation radius :math:`r`.
        phase_radius: Heisenberg-lattice phase-ramp radius :math:`m`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        translation_radius: int = 1,
        phase_radius: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_1 = _conv_block(in_channels, 16)
        self.encoder_2 = _conv_block(16, _BOTTLENECK_DIM)
        self.heisenberg = HeisenbergConv2d(
            in_channels=_BOTTLENECK_DIM,
            out_channels=_BOTTLENECK_DIM,
            spatial_size=(_BOTTLENECK_HW, _BOTTLENECK_HW),
            translation_radius=translation_radius,
            phase_radius=phase_radius,
        )
        self.decoder_2 = _conv_block(_BOTTLENECK_DIM, 16)
        self.decoder_1 = _conv_block(16 + 16, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        feat1 = self.encoder_1(x)
        pooled = self.down(feat1)
        feat2 = self.encoder_2(pooled)
        feat2_resized = nn.functional.interpolate(
            feat2, size=(_BOTTLENECK_HW, _BOTTLENECK_HW), mode="bilinear", align_corners=False
        )
        hb_out = self.heisenberg(feat2_resized)
        # Magnitude of the complex output is a real-valued feature map.
        magnitude = hb_out.abs()
        pooled_hw = (original_size[0] // 2, original_size[1] // 2)
        upsampled = nn.functional.interpolate(
            magnitude, size=pooled_hw, mode="bilinear", align_corners=False
        )
        dec2 = self.decoder_2(upsampled)
        dec2 = self.up(dec2)
        merged = torch.cat([dec2, feat1], dim=1)
        dec1 = self.decoder_1(merged)
        return self.head(dec1)
