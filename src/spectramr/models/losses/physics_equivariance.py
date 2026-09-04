"""Operator-equivariance and Bloch-equivariance losses (Ideas 3 and 6).

Both losses live in this module because they share the same mathematical
structure — a residual ``‖f(O(x)) − g(f(x))‖²`` where ``f`` is an
encoder / translator and ``O``, ``g`` are physics operators related by
a learned (Idea 6) or analytically known (Idea 3) intertwiner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


# `compatible_with=["image"]`: the tensors this grades are whatever the encoder
# emits, and an encoder that keeps spatial extent emits a per-pixel feature map
# in the image domain. `idea_6_phys_eq_ssl_brain` is that case -- its
# `enhanced_deep_unet` declares `output_domain='image'` and the strategy treats
# the feature map AS the latent. The loss never inspects a domain; it takes the
# pair `(f_θ(O(x)), g_φ(f_θ(x)))` the strategy hands it.
@register_loss("physics_equivariance", domain="latent", compatible_with=["image"])
class PhysicsEquivarianceLoss(nn.Module):
    """Operator-equivariant SSL loss (Idea 6).

    Given an encoder ``f_θ`` and a latent operator ``g_φ`` (per-operator
    matrix or MLP), require

        ‖ f_θ(O(x)) − g_φ ( f_θ(x) ) ‖²

    where ``O`` is a physics operator drawn from the framework's physics
    registry. The forward signature follows the
    :class:`IComputableLoss` protocol: it takes
    ``pred = f_θ(O(x))`` and ``target = g_φ(f_θ(x))`` and returns a dict.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        residual = F.mse_loss(pred, target)
        return {"loss": self.weight * residual, "physics_equivariance": residual}


@register_loss("bloch_equivariance_residual", domain="image")
class BlochEquivarianceResidual(nn.Module):
    """Bloch-equivariant cross-field-strength translation loss (Idea 3).

    ``‖ B_θ(F^B_ULF(q)) − F^B_HF(q) ‖²`` over parameter-atlas samples q.
    Caller supplies the two terms; the loss is just the weighted L2
    residual.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        residual = F.mse_loss(pred, target)
        return {
            "loss": self.weight * residual,
            "bloch_equivariance_residual": residual,
        }


__all__ = ["BlochEquivarianceResidual", "PhysicsEquivarianceLoss"]
