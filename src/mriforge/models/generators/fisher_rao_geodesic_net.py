"""Fisher-Rao geodesic cross-field translator (MICCAI MRIxFields2026, B-3.4).

A field-conditioned CNN predicts a per-pixel ANGULAR VELOCITY ``delta(x, b_s, b_t)`` on the
Bernoulli information manifold (each intensity ``x in [0,1]`` is a Bernoulli ``(x, 1-x)``).
The translated image is the Fisher-Rao GEODESIC SHOOTING step
``x_t = cos^2(arccos(sqrt(x)) + delta)`` (:mod:`mriforge.models.blocks.fisher_rao_sphere`),
i.e. moving along the great-circle of the information geometry — the canonical (Cencov)
invariant interpolation between field-indexed intensities. The output is in ``[0,1]`` BY
CONSTRUCTION (smoothly, without a clamp; the genuine distinction from a clamped Euclidean step).
NOTE the learned-velocity gradient ``-2 sqrt(x(1-x))`` vanishes at intensity boundaries — a
trainability/ablation-conditioning caveat documented in :mod:`mriforge.models.blocks.fisher_rao_sphere`.

``use_fisher_rao_geometry=False`` replaces the spherical geodesic step with a plain Euclidean
additive update ``(x + delta).clamp(0,1)`` — the one-knob ablation (isolates the
information-geometry vs Euclidean transport; both translate, so non-degenerate). The head is
zero-initialised so ``delta=0`` at init -> the map is the identity (a sane near-identity start,
config-invariant since ``delta`` is an additive angle).

Magnitude-only (1->1); both ``field_strength`` and ``field_strength_target`` are REQUIRED
forward kwargs (no default) so the Tier-2 probe injects them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from mriforge.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from mriforge.models.blocks.fisher_rao_sphere import fisher_rao_geodesic_step
from mriforge.models.registry import register_model


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(name="fisher_rao_geodesic_net", training_mode="fisher_rao_geodesic")
class FisherRaoGeodesicNet(nn.Module):
    """Per-pixel Fisher-Rao geodesic-shooting cross-field translator (B-3.4)."""

    # delta=0 at init -> identity (near-identity start); the identity-collapse probe heuristic
    # would false-positive, so suppress it (the input-invariance #20 check stays active —
    # the map is genuinely input-dependent). dead_loss is also suppressed: the real objective
    # is computed by FisherRaoGeodesicStrategy (compute_fisher_rao_loss), NOT the configured
    # placeholder image_losses, and the near-identity init drives that placeholder l1 to ~0 on
    # the probe's phantom — a false dead-loss verdict for an arm that trains on the cluster.
    synthetic_forward_probe_skip: ClassVar[set[str]] = {"identity_collapse", "dead_loss"}

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_blocks: int = 3,
        field_hidden: int = 32,
        use_fisher_rao_geometry: bool = True,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"FisherRaoGeodesicNet is magnitude-only (1->1); got in={in_channels}, out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        self.use_fisher_rao_geometry = bool(use_fisher_rao_geometry)
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        self.width = int(width)
        self.lift = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.ModuleList(_conv_block(width, width) for _ in range(n_blocks))
        # Per-block FiLM (gamma, beta) from (log10 b_s, log10 b_t) [dim 2], plus a contrast
        # one-hot appended when conditioning is on; zero-init -> identity FiLM.
        cond_in = contrast_sequence_dim(2, self.num_contrasts, self.use_contrast_conditioning)
        self.field_mlp = nn.Sequential(
            nn.Linear(cond_in, field_hidden),
            nn.SiLU(),
            nn.Linear(field_hidden, 2 * width * n_blocks),
        )
        nn.init.zeros_(self.field_mlp[-1].weight)
        nn.init.zeros_(self.field_mlp[-1].bias)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)  # delta=0 at init -> identity geodesic step
        nn.init.zeros_(self.head.bias)
        self.n_blocks = int(n_blocks)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        field_strength_target: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("FisherRaoGeodesicNet expects magnitude (real) input.")
        b_s = torch.log10(field_strength.reshape(-1, 1).float().clamp_min(1e-3))
        b_t = torch.log10(field_strength_target.reshape(-1, 1).float().clamp_min(1e-3))
        # Append the contrast one-hot to the (b_s, b_t) field pair when on (#15/#9).
        cond = build_contrast_sequence(
            torch.cat([b_s, b_t], dim=1),
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=x.shape[0],
            device=x.device,
            dtype=b_s.dtype,
        )
        film = self.field_mlp(cond).reshape(x.shape[0], self.n_blocks, 2, self.width)
        h = self.lift(x)
        for i, block in enumerate(self.blocks):
            h = block(h)
            gamma = (1.0 + film[:, i, 0])[:, :, None, None]
            beta = film[:, i, 1][:, :, None, None]
            h = gamma * h + beta
        delta = self.head(h)  # per-pixel angular velocity
        if self.use_fisher_rao_geometry:
            return fisher_rao_geodesic_step(x, delta)  # bounded [0,1] by construction
        return (x + delta).clamp(0.0, 1.0)  # Euclidean ablation
