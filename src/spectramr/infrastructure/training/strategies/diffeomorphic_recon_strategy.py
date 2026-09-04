"""Diffeomorphic-Registration-Coupled Reconstruction (ULF-PR-8 / II.5).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-8.

Joint reconstruction + intra-volume diffeomorphic registration via
LDDMM-style stationary velocity fields ``v`` and the exponential map
``φ = exp(v)`` computed by scaling-and-squaring.  This keeps the
recovered warp guaranteed-invertible (no folds) by construction.

Loss = parent reconstruction fidelity
     + ``λ_smooth · ‖∇v‖²``       (smoothness)
     + ``λ_fold  · max(-det J, 0)``  (hinge penalty on folds)
     + ``λ_warp  · ‖x ∘ φ⁻¹ - target‖₁``  (warped-image fidelity).

Public methods :meth:`exp_velocity` and :meth:`apply_warp` are exposed
for downstream pipelines (Phase-2 ``SeverelyWarpedPipeline``) to use
the same primitives at inference time.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy


class DiffeomorphicReconStrategy(ReconstructionTrainingStrategy):
    """Recon + diffeomorphic registration via scaling-and-squaring."""

    n_squarings: int = 6
    lambda_smooth: float = 1e-3
    lambda_fold: float = 1e-2
    lambda_warp: float = 1.0

    @staticmethod
    def _identity_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        return torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    def exp_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Compute φ = exp(v) by scaling-and-squaring.

        v: (B, 2, H, W).  Returns sampling grid (B, H, W, 2) suitable
        for ``F.grid_sample``.
        """
        B, _, H, W = v.shape
        scaled = v / (2**self.n_squarings)  # (B, 2, H, W)
        grid_id = self._identity_grid(B, H, W, v.device)  # (B, H, W, 2)
        grid = grid_id + scaled.permute(0, 2, 3, 1) * 0.5  # initial half-step
        for _ in range(self.n_squarings):
            # φ ← φ ∘ φ : sample current grid through itself.
            sampled = F.grid_sample(
                grid.permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            grid = grid + (sampled - grid_id)
        return grid

    @staticmethod
    def apply_warp(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    @staticmethod
    def _jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
        d_dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        d_dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        H, W = flow.shape[-2] - 1, flow.shape[-1] - 1
        d_dx = d_dx[:, :, :H, :W]
        d_dy = d_dy[:, :, :H, :W]
        return (1 + d_dx[:, 0]) * (1 + d_dy[:, 1]) - d_dx[:, 1] * d_dy[:, 0]

    @staticmethod
    def _smoothness(v: torch.Tensor) -> torch.Tensor:
        gx = v[:, :, :, 1:] - v[:, :, :, :-1]
        gy = v[:, :, 1:, :] - v[:, :, :-1, :]
        return gx.pow(2).mean() + gy.pow(2).mean()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (2026-05-17 round 5): see mri_slam_strategy.py for the
        # rationale. Orchestrator kwargs → canonical signature + helper.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base

        v = batch.get("velocity_field")
        if v is None:
            v = batch.get("deformation_field")
            if v is None:
                return base

        smooth = self.lambda_smooth * self._smoothness(v)
        fold = self.lambda_fold * F.relu(-self._jacobian_determinant(v)).mean()

        base["loss_velocity_smoothness"] = smooth
        base["loss_diffeomorphism_folding"] = fold

        warp_loss = v.new_zeros(())
        moving = pick_present(batch.get("moving"), batch.get("prediction"))
        target = batch.get("target")
        if moving is not None and target is not None:
            grid = self.exp_velocity(v)
            warped = self.apply_warp(moving, grid)
            warp_loss = self.lambda_warp * (warped - target).abs().mean()
            base["loss_warped_fidelity"] = warp_loss

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + smooth + fold + warp_loss
                break
        return base
