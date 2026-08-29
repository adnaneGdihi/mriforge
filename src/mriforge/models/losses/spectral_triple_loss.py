r"""Connes spectral-triple ULF→HF intertwining loss.

Implements the Morita-morphism training objective

.. math::

    \mathcal{L} = \|U\mathbf{x}-\mathbf{x}^{\rm HF}\|^2 + \lambda \|U D_{B_0^{\rm ULF}} - D_{B_0^{\rm HF}} U\|^2,

where the first term is the standard reconstruction residual and the
second is the operator-norm penalty enforcing that :math:`U` is a unitary
intertwiner of Dirac operators. See
:mod:`mriforge.infrastructure.physics.spectral_triple` for the underlying
algebraic primitives.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.spectral_triple import discrete_dirac_2d
from mriforge.models.losses.registry import register_loss


@register_loss(
    name="spectral_triple_intertwining",
    aliases=["connes_intertwining", "morita_morphism"],
    domain="image",
)
class SpectralTripleIntertwiningLoss(nn.Module):
    r"""Reconstruction + Dirac-intertwining penalty for ULF→HF translation.

    The loss expects the *predicted HF spinor* :math:`U\psi^{\rm ULF}` as
    ``prediction`` and the *ground-truth HF spinor* :math:`\psi^{\rm HF}` as
    ``target``. The ULF source is supplied as ``ulf_state`` in ``kwargs``
    (defaults to ``target`` for in-place testing).

    Args:
        intertwining_weight: Coefficient :math:`\lambda` on the Dirac
            commutator residual. Default 1.0.
        b0_ulf: ULF field strength (Tesla). Default 0.3.
        b0_hf: HF field strength (Tesla). Default 1.5.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        intertwining_weight: float = 1.0,
        b0_ulf: float = 0.3,
        b0_hf: float = 1.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if intertwining_weight < 0:
            raise ValueError("intertwining_weight must be >= 0")
        if b0_ulf <= 0 or b0_hf <= 0:
            raise ValueError("field strengths must be positive")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be mean/sum, got {reduction!r}")
        self.intertwining_weight = intertwining_weight
        self.b0_ulf = b0_ulf
        self.b0_hf = b0_hf
        self.reduction = reduction

    def _to_spinor(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-3] == 2:
            return x
        if x.shape[-3] == 1:
            return torch.cat([x, torch.zeros_like(x)], dim=-3)
        raise ValueError(f"Expected 1 or 2 channels on axis -3; got {x.shape[-3]}.")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        ulf_state = kwargs.get("ulf_state", target)
        if isinstance(ulf_state, torch.Tensor):
            ulf_spinor = self._to_spinor(ulf_state)
        else:
            ulf_spinor = self._to_spinor(target)

        pred_spinor = self._to_spinor(prediction)
        target_spinor = self._to_spinor(target)

        reduce = torch.mean if self.reduction == "mean" else torch.sum
        recon = reduce((pred_spinor - target_spinor).abs().pow(2))

        # Morita-intertwining residual: enforce that the Dirac action on the
        # *predicted* HF spinor matches the Dirac action on the *true* HF
        # spinor — i.e. that the network's translation respects the operator
        # algebraic structure on both sides. This is the practical surrogate
        # for ||U D_ULF - D_HF U||² when U itself is the network.
        d_target = discrete_dirac_2d(target_spinor, field_strength=self.b0_hf)
        d_pred = discrete_dirac_2d(pred_spinor, field_strength=self.b0_hf)
        intertwine_hf = reduce((d_pred - d_target).abs().pow(2))
        # Connes-distance cross-field penalty: the Dirac action on the ULF
        # input lower-bounds the Dirac action on the prediction (modulo the
        # field-strength scaling factor).
        d_ulf = discrete_dirac_2d(ulf_spinor, field_strength=self.b0_ulf)
        field_ratio = self.b0_hf / self.b0_ulf
        intertwine_cross = reduce((d_pred - field_ratio * d_ulf).abs().pow(2))

        return recon + self.intertwining_weight * (intertwine_hf + 0.1 * intertwine_cross)
