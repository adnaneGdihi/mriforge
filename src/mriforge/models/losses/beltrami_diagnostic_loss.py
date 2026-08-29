"""Beltrami coefficient diagnostic + (optional) regularizer.

For a planar map :math:`f = u + i v`, the Beltrami coefficient is

.. math::

    \\mu(f)
        = \\frac{\\partial f / \\partial \\bar z}{\\partial f / \\partial z}
        = \\frac{(u_x - v_y) + i (v_x + u_y)}
                {(u_x + v_y) + i (v_x - u_y)}.

The map is quasiconformal iff :math:`\\| \\mu \\|_\\infty < 1`. The
loss returns :math:`\\| \\mu \\|_\\infty` (or its squared mean) as a
differentiable diagnostic.

Per plan §12.2 — *"reframe QC as a diagnostic, not a guarantee"* —
the v0 YAML attaches this with ``weight = 0`` so it is logged only.
P2 promotes it to a soft regularizer once the training trajectory
is stable.

Input convention:

- 2D: ``(B, 2, H, W)`` interpreted as ``u`` and ``v`` channels.
- 3D: ``(B, 2, D, H, W)`` — Beltrami is computed per axial slice
  (the natural QC notion in 3D requires a Frenet frame; per-slice
  is the standard MRI-distortion proxy).

Fallback signature ``forward(pred, target)`` treats the residual
``pred - target`` as a 2-channel displacement field if it has 2
channels; otherwise raises ``ValueError``.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.losses.registry import register_loss


@register_loss(
    name="beltrami_diagnostic",
    aliases=["BeltramiDiagnostic", "beltrami"],
)
class BeltramiDiagnosticLoss(nn.Module):
    """Differentiable Beltrami coefficient ``||mu||`` for a (u, v) field.

    Args:
        reduction: ``"infty"`` returns the per-batch maximum
            ``||mu||_inf``; ``"mean"`` returns the per-pixel mean
            ``|mu|``; ``"penalty"`` returns the soft penalty
            ``relu(|mu| - 1).pow(2).mean()`` which is zero exactly
            when the map is quasiconformal.
        eps: stabilizer for the denominator.
    """

    def __init__(
        self,
        reduction: Literal["infty", "mean", "penalty"] = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if reduction not in {"infty", "mean", "penalty"}:
            raise ValueError(
                f"reduction must be one of {{infty, mean, penalty}}, got {reduction!r}"
            )
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        displacement: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Beltrami diagnostic.

        Pass either an explicit ``displacement`` tensor (preferred),
        or rely on the standard ``(pred, target)`` convention which
        uses the residual ``pred - target`` as the displacement
        field — only valid when both are 2-channel (u, v).
        """
        field = self._resolve_field(pred, target, displacement)
        # Now `field` is (B, 2, H, W) for 2D or (B, 2, D, H, W) for 3D.
        if field.ndim == 4:
            mu_abs = self._beltrami_2d(field)
        elif field.ndim == 5:
            # Per-slice Beltrami: vectorize batch and depth dims.
            B, two, D, H, W = field.shape
            assert two == 2
            stacked = field.permute(0, 2, 1, 3, 4).reshape(B * D, 2, H, W)
            mu_abs = self._beltrami_2d(stacked)
        else:
            raise ValueError(
                f"Beltrami input must be 4D (B,2,H,W) or 5D (B,2,D,H,W), got {field.ndim}D"
            )

        if self.reduction == "infty":
            # Per-element max — keep grad via amax.
            return mu_abs.amax()
        if self.reduction == "mean":
            return mu_abs.mean()
        # "penalty": softplus-style hinge above the QC bound.
        return torch.relu(mu_abs - 1.0).pow(2).mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None,
        displacement: torch.Tensor | None,
    ) -> torch.Tensor:
        if displacement is not None:
            return displacement
        if target is None:
            return pred
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have the same shape when displacement is None "
                f"(got {tuple(pred.shape)} vs {tuple(target.shape)})"
            )
        if pred.shape[1] != 2:
            raise ValueError(
                "Beltrami fallback (pred, target) requires 2-channel inputs "
                f"interpretable as (u, v); got {pred.shape[1]} channels"
            )
        return pred - target

    def _beltrami_2d(self, field: torch.Tensor) -> torch.Tensor:
        """Compute |mu(f)| pointwise for a (B, 2, H, W) field."""
        u = field[:, 0]
        v = field[:, 1]
        # Use central differences via roll; pad-by-replicate at edges
        # to avoid spurious wrap-around (we discard the last row/col).
        u_x = self._dx(u)
        u_y = self._dy(u)
        v_x = self._dx(v)
        v_y = self._dy(v)

        # mu = ( (u_x - v_y) + i(v_x + u_y) ) / ( (u_x + v_y) + i(v_x - u_y) )
        num_re = u_x - v_y
        num_im = v_x + u_y
        den_re = u_x + v_y
        den_im = v_x - u_y

        num_sq = num_re.pow(2) + num_im.pow(2)
        den_sq = den_re.pow(2) + den_im.pow(2) + self.eps
        return (num_sq / den_sq).sqrt()

    @staticmethod
    def _dx(t: torch.Tensor) -> torch.Tensor:
        # Central finite difference along the W axis with replicate
        # padding so the output shape matches the input.
        tp = F.pad(t.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(1)
        return 0.5 * (tp[..., :, 2:] - tp[..., :, :-2])

    @staticmethod
    def _dy(t: torch.Tensor) -> torch.Tensor:
        # Central finite difference along the H axis with replicate
        # padding so the output shape matches the input.
        tp = F.pad(t.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
        return 0.5 * (tp[..., 2:, :] - tp[..., :-2, :])


__all__ = ["BeltramiDiagnosticLoss"]
