r"""Helmholtz-Hodge decomposition loss for motion / displacement fields.

The ``hodge_motion`` strategy advertises Helmholtz-Hodge mathematics but
historically routed to a generic reconstruction strategy with vanilla
pixel losses (a façade). This loss closes that gap by encoding the Hodge
*structure* of a motion field directly into the objective, consuming the
existing primitive in
:mod:`spectramr.infrastructure.physics.helmholtz_hodge`.

A vector (motion) field :math:`\mathbf{v}` splits :math:`L^2`-orthogonally
into a curl-free (irrotational, gradient) part and a divergence-free
(solenoidal, rotational) part:

.. math::

    \mathbf{v} = \underbrace{\nabla\phi}_{\text{irrotational}}
               + \underbrace{\mathbf{v}_{\rm sol}}_{\text{solenoidal}},
    \qquad \langle\nabla\phi,\mathbf{v}_{\rm sol}\rangle_{L^2}=0.

For an *incompressible* motion field (e.g. cardiac contraction) the
irrotational part should vanish: penalising the divergence of the
prediction enforces this inductive bias. We additionally match the
decomposed components of the prediction against those of the target so
the network learns the correct partition of motion energy rather than
just the raw displacement.

Treats two channels of the ``[B, C, H, W]`` tensor as the planar
:math:`(u, v)` vector field. The 3D primitive is invoked with a degenerate
depth axis (``D=1``), which makes all :math:`\partial/\partial z` terms
vanish so the 3D divergence collapses exactly to the 2D divergence
:math:`\partial_y u + \partial_x v`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.helmholtz_hodge import (
    divergence,
    helmholtz_hodge_decompose,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="hodge_decomposition",
    aliases=["hodge_motion", "helmholtz_hodge"],
    domain="image",
)
class HodgeDecompositionLoss(nn.Module):
    r"""Helmholtz-Hodge structural loss on a 2D motion / displacement field.

    The first two channels of ``prediction`` / ``target`` are interpreted
    as the planar vector field :math:`(u, v)` (channel 0 = :math:`v_y`
    indexed along ``H``, channel 1 = :math:`v_x` indexed along ``W``).

    The total loss is

    .. math::

        \mathcal{L} = \lambda_{\rm div}\,\| \nabla\!\cdot\mathbf{v}_{\rm pred}
                                     - \nabla\!\cdot\mathbf{v}_{\rm tgt} \|
                    + \lambda_{\rm comp}\,\big(
                        \| \mathbf{v}^{\rm pred}_{\rm irrot}
                         - \mathbf{v}^{\rm tgt}_{\rm irrot}\|
                      + \| \mathbf{v}^{\rm pred}_{\rm sol}
                         - \mathbf{v}^{\rm tgt}_{\rm sol}\|
                      \big),

    where the irrotational / solenoidal parts come from
    :func:`spectramr.infrastructure.physics.helmholtz_hodge.helmholtz_hodge_decompose`
    and the divergence from :func:`...helmholtz_hodge.divergence`. The
    divergence term penalises spurious compressibility; the component term
    forces the prediction to reproduce the target's *partition* of motion
    energy, not merely its raw displacement.

    Args:
        divergence_weight: :math:`\lambda_{\rm div}` on the
            divergence-discrepancy (incompressibility) term. Default 1.0.
        component_match_weight: :math:`\lambda_{\rm comp}` on the
            decomposed-component matching term. Default 1.0.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        divergence_weight: float = 1.0,
        component_match_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if divergence_weight < 0:
            raise ValueError("divergence_weight must be >= 0")
        if component_match_weight < 0:
            raise ValueError("component_match_weight must be >= 0")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.divergence_weight = divergence_weight
        self.component_match_weight = component_match_weight
        self.reduction = reduction

    @staticmethod
    def _to_3d_vector(field_bchw: torch.Tensor) -> torch.Tensor:
        r"""Map ``[B, 2, H, W]`` planar field to a batch of ``(3, 1, H, W)``.

        Builds the 3-component vector ``(vz, vy, vx) = (0, u, v)`` on a
        single-slice (``D=1``) grid. With ``D=1`` every
        :math:`\partial/\partial z` central difference is zero, so the 3D
        divergence/decomposition collapses to the intended 2D operator.

        Returns:
            Tensor of shape ``[B, 3, 1, H, W]``.
        """
        u = field_bchw[:, 0]  # vy
        v = field_bchw[:, 1]  # vx
        vz = torch.zeros_like(u)
        vec = torch.stack([vz, u, v], dim=1)  # [B, 3, H, W]
        return vec.unsqueeze(2)  # [B, 3, 1, H, W]

    def _reduce(self, t: torch.Tensor) -> torch.Tensor:
        return torch.mean(t) if self.reduction == "mean" else torch.sum(t)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share shape; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError(
                f"expected a [B, C, H, W] motion field; got {tuple(prediction.shape)}."
            )
        if prediction.shape[1] < 2:
            raise ValueError(
                "Helmholtz-Hodge requires >= 2 channels for the (u, v) "
                f"vector field; got C={prediction.shape[1]}."
            )

        pred_vec = self._to_3d_vector(prediction[:, :2])  # [B, 3, 1, H, W]
        tgt_vec = self._to_3d_vector(target[:, :2])

        total = prediction.new_zeros(())

        # --- Incompressibility / divergence-discrepancy term -------------
        if self.divergence_weight > 0:
            # `divergence` accepts a leading batch axis: (..., 3, D, H, W).
            div_pred = divergence(pred_vec)  # [B, 1, H, W]
            div_tgt = divergence(tgt_vec)
            total = total + self.divergence_weight * self._reduce((div_pred - div_tgt).abs())

        # --- Decomposed-component matching term --------------------------
        if self.component_match_weight > 0:
            comp_pred_irrot = []
            comp_pred_sol = []
            comp_tgt_irrot = []
            comp_tgt_sol = []
            # `helmholtz_hodge_decompose` is defined per (3, D, H, W) field.
            for b in range(pred_vec.shape[0]):
                pi, ps = helmholtz_hodge_decompose(pred_vec[b])
                ti, ts = helmholtz_hodge_decompose(tgt_vec[b])
                comp_pred_irrot.append(pi)
                comp_pred_sol.append(ps)
                comp_tgt_irrot.append(ti)
                comp_tgt_sol.append(ts)
            pred_irrot = torch.stack(comp_pred_irrot, dim=0)
            pred_sol = torch.stack(comp_pred_sol, dim=0)
            tgt_irrot = torch.stack(comp_tgt_irrot, dim=0)
            tgt_sol = torch.stack(comp_tgt_sol, dim=0)

            comp = self._reduce((pred_irrot - tgt_irrot).abs()) + self._reduce(
                (pred_sol - tgt_sol).abs()
            )
            total = total + self.component_match_weight * comp

        return total
