r"""MCGI -- Monotone-Contrast-Group-Invariant encoder.

An encoder :math:`\Phi = \Psi \circ R` whose invariance to monotone intensity
remaps is guaranteed *by construction* rather than learned in expectation
(contrast the randomization-trained approximate invariance of SynthSeg /
SynthMorph). :math:`R` is the per-image empirical-CDF rank transform
(:mod:`mriforge.models.blocks.rank_normalize`), which is exactly invariant to the
group :math:`G_+` of strictly increasing intensity maps; :math:`\Psi` is any
backbone.

For order-reversing contrast pairs (T1w vs T2w) adjoin the involution
:math:`\sigma:\mathbf x\mapsto-\mathbf x` and symmetrize,

.. math::

   \Phi_{\mathrm{sym}}(\mathbf x)
     = \tfrac12\bigl(\Psi(R(\mathbf x)) + \Psi(R(-\mathbf x))\bigr),

a :math:`\mathbb Z_2`-invariant section of the quotient, so :math:`\Phi_{\mathrm
{sym}}` is exactly invariant on the full group :math:`G_+ \rtimes \mathbb Z_2`.

Scope (honest): invariant features discard absolute-contrast information, so MCGI
is for anatomy-defined tasks (segmentation, registration) and not for contrast
quantification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.rank_normalize import RankNormalize2d
from mriforge.models.registry import register_model


def _conv_backbone(
    in_channels: int, hidden_channels: int, out_channels: int, depth: int
) -> nn.Sequential:
    """A small same-resolution conv stack standing in for :math:`\\Psi`.

    The invariance guarantee is independent of the backbone (it sits *behind*
    the rank transform), so a plain conv stack is sufficient and honest.
    """
    layers: list[nn.Module] = []
    c_in = in_channels
    for _ in range(max(depth - 1, 0)):
        layers += [nn.Conv2d(c_in, hidden_channels, 3, padding=1), nn.GELU()]
        c_in = hidden_channels
    layers += [nn.Conv2d(c_in, out_channels, 3, padding=1)]
    return nn.Sequential(*layers)


@register_model(
    name="mcgi_encoder",
    training_mode="supervised",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=False,
)
class MCGIEncoder(nn.Module):
    r"""Monotone-contrast-group-invariant encoder :math:`\Phi = \Psi \circ R`.

    Args:
        in_channels: Input image channels.
        hidden_channels: Backbone hidden width.
        out_channels: Output feature channels.
        depth: Number of conv blocks in the backbone.
        symmetrize_order_reversal: Pool over :math:`\sigma:\mathbf x\mapsto-\mathbf
            x` to extend invariance to :math:`G_+\rtimes\mathbb Z_2`.
        soft_rank_temperature: Soft-rank temperature used in training mode.
        hard_rank_eval: Use the exact hard rank at inference (the guarantee).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 1,
        depth: int = 3,
        *,
        symmetrize_order_reversal: bool = True,
        soft_rank_temperature: float = 0.05,
        hard_rank_eval: bool = True,
    ) -> None:
        super().__init__()
        self.symmetrize_order_reversal = bool(symmetrize_order_reversal)
        self.rank = RankNormalize2d(temperature=soft_rank_temperature, hard_eval=hard_rank_eval)
        self.backbone = _conv_backbone(in_channels, hidden_channels, out_channels, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(self.rank(x))
        if self.symmetrize_order_reversal:
            feat_rev = self.backbone(self.rank(-x))
            feat = 0.5 * (feat + feat_rev)
        return feat


__all__ = ["MCGIEncoder"]
