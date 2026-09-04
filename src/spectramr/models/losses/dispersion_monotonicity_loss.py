r"""Dispersion-monotonicity penalty for DL-BAE (M4).

Enforces the physical constraint :math:`\partial T_1/\partial B_0 \ge 0` on the
decoded relaxation maps. BPP relaxation theory makes :math:`T_1` non-decreasing
in field over the physiological :math:`\tau_c` range: as :math:`\omega_0\tau_c`
grows, the spectral density :math:`J(\omega_0;\tau_c)` falls, the longitudinal
rate falls, and :math:`T_1` rises.

The fit can still wander into a non-monotone region under noise -- especially
with :math:`P>1` pools, where two pools can trade off against each other -- and
a non-monotone :math:`T_1(B_0)` is not a slightly-worse fit but a physically
impossible one that invalidates any cross-field extrapolation. So it is
penalised, not ignored.

This is a **soft projection**: a hinge on the negative part of the finite
difference along the field axis, zero exactly when the constraint holds, so it
adds no gradient to an already-physical solution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

__all__ = ["DispersionMonotonicity"]


@register_loss(name="dispersion_monotonicity", domain="physics")
class DispersionMonotonicity(nn.Module):
    r"""Hinge penalty on :math:`\partial T_1/\partial B_0 < 0`.

    Args:
        tolerance: Slack (seconds) before a decrease is penalised; absorbs
            floating-point noise without excusing a real violation.
        squared: Square the hinge (smoother gradient near the boundary).
    """

    def __init__(self, tolerance: float = 1e-6, squared: bool = False) -> None:
        super().__init__()
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative; got {tolerance}.")
        self.tolerance = float(tolerance)
        self.squared = bool(squared)

    def forward(self, t1_maps: torch.Tensor, fields: torch.Tensor | None = None) -> torch.Tensor:
        r"""Penalise field-wise decreases in :math:`T_1`.

        Args:
            t1_maps: :math:`T_1` maps ``[B, M, H, W]`` (seconds), with the field
                axis **sorted ascending** unless ``fields`` is given.
            fields: Optional field strengths ``[M]``; when supplied, ``t1_maps``
                is sorted by it first, so the caller need not pre-sort.

        Returns:
            Scalar penalty; exactly zero when :math:`T_1` is non-decreasing.

        Raises:
            ValueError: when ``t1_maps`` is not 4-D or ``fields`` mismatches it.
        """
        if t1_maps.ndim != 4:
            raise ValueError(
                f"dispersion_monotonicity expects [B, M, H, W]; got {tuple(t1_maps.shape)}."
            )
        if fields is not None:
            if fields.numel() != t1_maps.shape[1]:
                raise ValueError(
                    f"fields has {fields.numel()} entries but t1_maps carries "
                    f"{t1_maps.shape[1]} fields."
                )
            order = torch.argsort(fields.flatten())
            t1_maps = t1_maps.index_select(1, order.to(t1_maps.device))
        if t1_maps.shape[1] < 2:
            # A single field carries no monotonicity information. Return a
            # zero that is still connected to the graph, so a strategy summing
            # this term keeps a valid backward pass.
            return t1_maps.sum() * 0.0

        decrease = torch.relu(-(t1_maps.diff(dim=1) + self.tolerance))
        if self.squared:
            decrease = decrease.pow(2)
        return decrease.mean()
