"""Koopman temporal motion encoder (2026-05-26).

Lifts the per-line rigid-pose sequence ``θ(τ)`` ``[B, 3, N_lines]`` into a
latent whose time evolution is governed by a *linear* Koopman operator
``z_{τ+1} = K z_τ``. Linear latent dynamics are a strong inductive bias for the
quasi-periodic motion (respiratory/cardiac) the digital twin generates, and
the operator ``K`` is directly inspectable (eigenvalues ↔ motion frequencies).

Registered as a conditioner under ``"motion_koopman"`` reading the same
``motion_pose`` source as :class:`MotionPoseEncoder`, so an experiment chooses
either the simple pooled-pose embedding or the Koopman temporal one in YAML.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.conditioning.context import ConditioningContext
from mriforge.models.conditioning.encoders import ConditionEncoder, register_conditioner


@register_conditioner("motion_koopman")
class KoopmanMotionEncoder(ConditionEncoder):
    """Pose-sequence → Koopman latent embedding.

    Args:
        embed_dim: latent (and output embedding) dimension.
        pose_dim: per-line pose DoF (3 for rigid: ``Δx, Δy, Δθ``).
    """

    source = "motion_pose"

    def __init__(self, embed_dim: int = 128, pose_dim: int = 3) -> None:
        super().__init__(embed_dim)
        self.pose_dim = int(pose_dim)
        self.enc = nn.Sequential(
            nn.Linear(self.pose_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        # Koopman operator — linear, no bias (so advance is a genuine linear map).
        self.K = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def encode(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """``[B, 3, N]`` → per-line latent sequence ``[B, N, embed_dim]``."""
        return self.enc(pose_seq.transpose(1, 2).float())

    def advance(self, z: torch.Tensor) -> torch.Tensor:
        """One Koopman step ``z ↦ K z`` (linear)."""
        return self.K(z)

    def rollout(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        """Advance ``z0`` ``steps`` times → ``[B, steps, embed_dim]``."""
        zs = []
        z = z0
        for _ in range(steps):
            z = self.K(z)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def forward(self, ctx: ConditioningContext) -> torch.Tensor:
        pose = self._require(ctx)  # [B, 3, N]
        return self.encode(pose).mean(dim=1)


__all__ = ["KoopmanMotionEncoder"]
