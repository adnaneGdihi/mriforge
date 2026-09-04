r"""Charbonnier (smooth-L1) loss — robust differentiable :math:`\ell_1`.

Defined as

.. math::
    \mathcal{L}_{\text{char}} = \sum \sqrt{(\hat{\mathbf{x}}-\mathbf{x})^2+\varepsilon^2}

The square-root keeps the gradient bounded near zero (unlike pure L1) and
behaves like L2 for small residuals, like L1 for large ones — i.e. an
edge-preserving robust regression term. Standard pixel loss in image
restoration, super-resolution, and unrolled MRI reconstruction.

Reference:
    P. Charbonnier *et al.*, "Two deterministic half-quadratic regularization
    algorithms for computed imaging," *Proc. IEEE ICIP*, 1994.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="charbonnier", aliases=["CharbonnierLoss"], domain="agnostic")
class CharbonnierLoss(nn.Module):
    r"""Charbonnier penalty: :math:`\sqrt{r^2 + \varepsilon^2}`.

    **DOMAIN**: AGNOSTIC — works on any tensor (image, k-space mag, latent).

    Args:
        epsilon: Smoothing constant. Smaller → closer to pure L1; larger →
            closer to L2. Default 1e-3 (standard in SR literature).
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        diff = prediction - target
        per_voxel = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        if self.reduction == "sum":
            return per_voxel.sum()
        if self.reduction == "none":
            return per_voxel
        return per_voxel.mean()
