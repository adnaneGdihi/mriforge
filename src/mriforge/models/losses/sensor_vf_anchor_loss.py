r"""Sensor-VF anchor loss (idea 9 §9.3).

Soft anchor that penalises divergence between the *sensor-network's*
SE(3) pose estimate :math:`\hat\zeta_\theta` and the *virtual-fiducial*
pose :math:`T_{VF}` derived from the existing VF infrastructure.

The proposal phrases the loss as a weighted Mahalanobis-like
distance: pose components weighted by the inverse of their reported
sensor-fusion variance, so noisy sensor readings receive less
anchoring force.

.. math::

    \mathcal{L}_{VF\text{-anchor}} = \lambda \,
        (\hat\zeta_\theta - T_{VF})^\top \Sigma^{-1} (\hat\zeta_\theta - T_{VF})

When ``Sigma`` is omitted the loss collapses to a plain MSE — useful
for early-training warm-up where the variance estimate is unreliable.

Reference
---------
Plan §9.3 of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.losses.registry import register_loss


@register_loss(
    name="sensor_vf_anchor",
    aliases=["SensorVFAnchorLoss", "vf_anchor"],
    domain="image",
)
class SensorVFAnchorLoss(nn.Module):
    r"""Mahalanobis-style anchor between sensor pose and VF pose.

    Args:
        weight: ``λ`` multiplier. Must be ≥ 0.
        eps: Tikhonov regulariser added to the covariance before inversion.

    Inputs to ``forward``:
        - ``sensor_pose``: ``[B, 6]`` predicted SE(3) pose
          ``(t_x, t_y, t_z, r_x, r_y, r_z)``.
        - ``vf_pose``: ``[B, 6]`` virtual-fiducial pose (no_grad target).
        - ``covariance``: optional ``[B, 6, 6]`` per-sample covariance.
          When ``None`` the loss reduces to plain MSE.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0; got {eps}")
        self.weight = float(weight)
        self.eps = float(eps)

    def forward(
        self,
        sensor_pose: torch.Tensor,
        vf_pose: torch.Tensor,
        covariance: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if sensor_pose.dim() != 2 or sensor_pose.shape[-1] != 6:
            raise ValueError(f"sensor_pose must be [B, 6]; got {tuple(sensor_pose.shape)}")
        if vf_pose.shape != sensor_pose.shape:
            raise ValueError(
                f"vf_pose shape {tuple(vf_pose.shape)} does not match "
                f"sensor_pose shape {tuple(sensor_pose.shape)}"
            )

        diff = sensor_pose - vf_pose.detach()  # VF is the anchor target.

        if covariance is None:
            return self.weight * (diff**2).mean()

        if covariance.dim() != 3 or covariance.shape[-2:] != (6, 6):
            raise ValueError(f"covariance must be [B, 6, 6]; got {tuple(covariance.shape)}")
        if covariance.shape[0] != sensor_pose.shape[0]:
            raise ValueError("Batch-size mismatch between sensor_pose and covariance.")

        # Per-sample Mahalanobis distance: dᵀ Σ⁻¹ d.
        eye = torch.eye(6, dtype=covariance.dtype, device=covariance.device)
        cov_reg = covariance + self.eps * eye
        info = torch.linalg.inv(cov_reg)
        # diff: [B, 6]; info: [B, 6, 6]
        scaled = torch.einsum("bij,bj->bi", info, diff)
        per_sample = (diff * scaled).sum(dim=-1)
        return self.weight * per_sample.mean()


__all__ = ["SensorVFAnchorLoss"]
