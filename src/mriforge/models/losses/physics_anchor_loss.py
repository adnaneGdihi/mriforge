r"""Physics-anchoring loss for the Neural-Operator Signal Equation (NOSE).

The anchor term :math:`\beta\lVert\mathcal G_\theta(\boldsymbol\xi,\boldsymbol
\varphi)-\mathcal A_{\mathrm{analytic}}(\boldsymbol\xi;\boldsymbol\varphi)\rVert^2`
ties the learned operator's output to the analytic signal equation. As
:math:`\beta\to\infty` the minimiser is :math:`\mathcal G_\theta\to\mathcal
A_{\mathrm{analytic}}`, recovering analytic synthetic MRI as the trusted-physics
limit and guaranteeing the operator never underperforms it. Where the analytic
model is known to be incomplete (magnetization transfer, B1 inhomogeneity,
partial volume), the schedule relaxes :math:`\beta` so the operator can absorb
the residual.

Reuses the analytic render via
:func:`mriforge.infrastructure.calibration.scores.physics_residual` (the same
Bloch SSOT used by PR-CC), so there is one rendering path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


@register_loss(name="physics_anchor", domain="image")
class PhysicsAnchorLoss(nn.Module):
    """Squared residual between an operator output and the analytic Bloch render.

    Args:
        weight: Static multiplier (the strategy schedules ``beta``).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """``beta * mean(|G(xi,phi) - A_analytic(xi;phi)|^2)``.

        Args:
            prediction: Operator output ``G(xi, phi)`` ``[B, 1, ...]``.
            target: Unused (kept for the loss contract; the anchor target is the
                analytic render built from ``context``).
            context: ``{"params": (rho,T1,T2) map, "acquisition": {TR,TE,TI},
                "contrast_type": str}``.

        Raises:
            ValueError: missing ``params`` / ``acquisition`` in ``context``.
        """
        if not context or "params" not in context or "acquisition" not in context:
            raise ValueError("physics_anchor needs context with 'params' and 'acquisition'.")
        # Lazy import: models/ must not import infrastructure/ at module scope
        # (layer-direction rule; calibration.scores is not physics-SSOT-exempt).
        from mriforge.infrastructure.calibration.scores import physics_residual

        # physics_residual(params, prediction) = |A_analytic(params) - prediction|.
        residual = physics_residual(
            context["params"],
            prediction,
            acquisition=context["acquisition"],
            contrast_type=context.get("contrast_type", "spin_echo"),
        )
        return self.weight * residual.pow(2).mean()


__all__ = ["PhysicsAnchorLoss"]
