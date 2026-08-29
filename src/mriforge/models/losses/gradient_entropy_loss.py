"""Gradient-entropy sharpness loss (image-domain autofocus metric).

The entropy of the per-image normalised gradient-magnitude distribution
(Atkinson et al., autofocus). A sharp image concentrates gradient energy at a
few edges (a sparse, low-entropy distribution); a blurred image spreads it
(higher entropy). So **lower is sharper** — the standard minimise convention.

Scope (exp_vf_35): this is a no-reference sharpness *diagnostic*, used under
``no_grad`` to report the recon-sharpness gain of the estimated trajectory vs
the nominal one. It is NOT a trajectory-training objective: the NUFFT backend
(torchkbnufft) does not propagate gradients to the trajectory coordinates, so a
through-NUFFT sharpness loss cannot train the GIRF parameters θ — supervising it
that way would be a silently-inert mechanism (pitfall #16). The trajectory model
is trained by the supervised Δk loss instead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import pad

from mriforge.models.losses.registry import register_loss

__all__ = ["GradientEntropyLoss"]


@register_loss(name="gradient_entropy", domain="image")
class GradientEntropyLoss(nn.Module):
    r"""Per-image gradient-magnitude entropy (lower = sharper).

    Args:
        eps: numerical floor for the log / normalisation.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"`` (per-image).
    """

    def __init__(self, eps: float = 1e-12, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,  # unused — no-reference metric
        **kwargs: Any,
    ) -> torch.Tensor:
        x = prediction.abs() if torch.is_complex(prediction) else prediction
        x = x.float()
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        gx = pad(gx, (0, 1, 0, 0))
        gy = pad(gy, (0, 0, 0, 1))
        g = torch.sqrt(gx * gx + gy * gy + self.eps)  # gradient magnitude
        b = g.shape[0]
        gf = g.reshape(b, -1)
        p = gf / gf.sum(dim=1, keepdim=True).clamp_min(self.eps)  # distribution per image
        entropy = -(p * (p + self.eps).log()).sum(dim=1)  # [B], lower = sharper
        if self.reduction == "sum":
            return entropy.sum()
        if self.reduction == "none":
            return entropy
        return entropy.mean()
