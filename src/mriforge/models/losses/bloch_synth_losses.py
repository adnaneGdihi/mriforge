"""Bloch-synthesis relaxometry losses (MICCAI MRIxFields2026, idea 2.1).

Two terms specific to the ``bloch_synth`` arm. Both are computed inline by
:class:`BlochSynthesisStrategy` (they need the encoder's parameter maps / the frozen
signal operator, not a plain ``(pred, target)`` pair), and registered here so they are
discoverable and unit-testable. The segmentation-consistency term is NOT defined here:
it reuses the existing ``segmentation_dice`` loss (pitfall #12 — one home per kind).
"""

from __future__ import annotations

import torch
from torch import nn

from mriforge.models.losses.registry import register_loss


@register_loss(
    name="dispersion_prior",
    domain="image",
    compatible_with=["bloch_synth"],
)
class DispersionPriorLoss(nn.Module):
    r"""Penalise the dispersion exponent outside the physiological envelope.

    ``mean( relu(lo - beta) + relu(beta - hi) )`` — zero when every voxel's
    :math:`\beta \in [lo, hi]` (Rooney et al.: :math:`\beta \in [0.3, 0.4]`), positive
    and linear outside. Keeps the learned power-law transport physical and keeps this
    term load-bearing (the encoder's ``beta`` head is bounded to a slightly wider
    envelope, so excursions into [lo-delta, lo) or (hi, hi+delta] are real and penalised).
    """

    def __init__(self, weight: float = 1.0, lo: float = 0.3, hi: float = 0.4) -> None:
        super().__init__()
        if not (lo < hi):
            raise ValueError(f"dispersion bounds require lo < hi; got ({lo}, {hi}).")
        self.weight = weight
        self.lo = lo
        self.hi = hi

    def forward(self, beta: torch.Tensor, **_: object) -> torch.Tensor:
        below = torch.relu(self.lo - beta)
        above = torch.relu(beta - self.hi)
        return self.weight * torch.mean(below + above)


@register_loss(
    name="bloch_source_consistency",
    domain="image",
    compatible_with=["bloch_synth"],
)
class BlochSourceConsistencyLoss(nn.Module):
    r"""Source-consistency :math:`\lVert C_s(\hat x) - y_s \rVert_1`.

    The estimated quantitative maps, rendered by the frozen SPGR signal operator back
    at the SOURCE field, must reproduce the source acquisition. The strategy supplies
    ``prediction = render(params, b_s)`` and ``target`` = the source image; this is the
    ``y = C_s(x)`` half of the identify-then-resynthesise loop that makes the inversion
    well-posed (Proposition 3). A thin L1, like ``latent_cycle``.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(prediction - target))
