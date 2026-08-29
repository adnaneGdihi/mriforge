"""Beltrami-parameterised adaptive space-filling-curve block (§1).

Warps the framework's fixed 2-D Hilbert curve by a learnable
quasi-conformal homeomorphism. A small head predicts a Beltrami
coefficient ``μ_θ`` (constrained to ``|μ| ≤ k < 1`` via
:func:`enforce_beltrami_constraint`); :class:`LinearBeltramiSolver`
turns it into a fold-free warp ``φ = z + h`` with bounded dilatation
``K = (1+k)/(1-k)``. The adaptive scan then orders grid cells by the
**base-Hilbert rank of their warped position** ``φ(z)``, so the
traversal bends toward content while staying locality-preserving (a
bounded-distortion map cannot scramble neighbouring ranks). When
``μ_θ ≡ 0`` the warp is the identity and the ordering is *exactly* the
fixed Hilbert curve — so the strategy strictly generalises a
fixed-Hilbert HSSC scan (pinned by
``test_beltrami_sfc_block_zero_mu_is_hilbert``).

Honest status (pitfall #16). The permutation is produced by an
``argsort`` and is therefore **not differentiable**: the reconstruction
loss cannot shape the curve, and the only gradient reaching ``μ_θ`` is
the strategy's ``|μ|²`` and trajectory regularisers — which *shrink*
the warp back toward the fixed Hilbert baseline. No fixed-Hilbert
control arm exists in the cohort, so "adaptive beats fixed" is not yet
attributable. A differentiable Gumbel-Sinkhorn alternative
(:class:`~mriforge.models.blocks.sinkhorn_sfc.SinkhornSFCLinearizer`) is
backlogged — see
``TODO/backlog_sfc_topology_inert_mechanisms_2026_07.md``.

Earlier this block sorted cells by ``φ.angle()`` — a single global
scalar that collapses 2-D locality (max L1 step ≈ 13 vs the Hilbert
curve's 1) and did *not* reduce to Hilbert at ``μ=0``, contradicting
the docstring's own (untested) consistency claim.

References:
    [1] L. V. Ahlfors, *Lectures on Quasiconformal Mappings*,
        AMS, 2006.
    [8] L. M. Lui et al., "Teichmüller mapping (T-map)", *SIAM J.
        Imag. Sci.*, 7(1), 2014.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.infrastructure.physics.conformal_geometry import (
    LinearBeltramiSolver,
    enforce_beltrami_constraint,
)
from mriforge.models.blocks.topology_linearizer import _hilbert_2d_indices


class BeltramiSFCBlock(nn.Module):
    """Learnable Beltrami-adaptive SFC over a 2-D grid.

    Args:
        grid_size: Square grid side. Must be a power of two
            (constraint from the Hilbert generator).
        in_channels: Number of conditioning channels fed to the
            μ head (typically the partial reconstruction or a
            content prior).
        hidden: Hidden width of the μ head.
        k_max: Upper bound on ``|μ|``. Smaller values make the
            adaptive curve closer to Hilbert; ``k_max → 1`` allows
            maximal warping (and risks numerical instability).
        solver: Optional pre-built solver; defaults to a fresh
            :class:`LinearBeltramiSolver`.
    """

    def __init__(
        self,
        *,
        grid_size: int,
        in_channels: int = 1,
        hidden: int = 32,
        k_max: float = 0.9,
        solver: LinearBeltramiSolver | None = None,
    ) -> None:
        super().__init__()
        if grid_size & (grid_size - 1):
            raise ValueError(f"grid_size must be a power of two, got {grid_size}")
        self.grid_size = grid_size
        self.k_max = k_max
        self.solver = solver or LinearBeltramiSolver(max_iter=20, tol=1e-3)

        # Tiny ConvNet that maps a content feature map to a complex μ field.
        self.mu_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),  # real + imag of μ
        )

        # Fixed Hilbert traversal indices on the grid, cached as a buffer.
        idx = torch.as_tensor(_hilbert_2d_indices(grid_size, grid_size), dtype=torch.long)
        self.register_buffer("hilbert_indices", idx, persistent=False)
        # Inverse permutation: hilbert_rank[g] = the step at which grid cell g is
        # visited along the Hilbert curve. Used to order cells by the rank of
        # their warped position (see forward).
        self.register_buffer("hilbert_rank", idx.argsort(), persistent=False)

    def _mu_from_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the constrained Beltrami coefficient from features."""
        out = self.mu_head(x)
        mu = enforce_beltrami_constraint(torch.complex(out[:, 0], out[:, 1]), k_max=self.k_max)
        return mu

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the adaptive SFC indices plus the underlying μ / φ fields.

        Args:
            features: ``[B, C, H, W]`` content tensor. ``H = W =
                grid_size`` is enforced.

        Returns:
            Dict with ``{"indices": [B, H*W] int64, "mu": [B, H, W],
            "phi": [B, H, W]}``. ``indices[b, k]`` is the flat grid
            index of the cell visited k-th in the adaptive scan
            (consumed by downstream Mamba scans); it is
            **non-differentiable** (produced by ``argsort``). ``mu`` and
            ``phi`` carry gradient and are exposed for the Beltrami /
            trajectory regularisers and diagnostics.
        """
        if features.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError(
                f"BeltramiSFCBlock got features of spatial shape "
                f"{tuple(features.shape[-2:])}; expected ({self.grid_size},)*2"
            )
        gs = self.grid_size
        n = gs * gs
        mu = self._mu_from_features(features)
        phi = self.solver(mu).phi  # φ = z + h on the canonical [-1, 1] grid

        # Warp each cell's position by φ and re-quantise to a grid cell.
        # φ.real is the warped x (column) coordinate, φ.imag the warped y (row).
        scale = 0.5 * (gs - 1)
        col = ((phi.real + 1.0) * scale).round().clamp_(0, gs - 1).long()
        row = ((phi.imag + 1.0) * scale).round().clamp_(0, gs - 1).long()
        warped_flat = (row * gs + col).view(features.shape[0], -1)  # [B, N]

        # Order cells by the base-Hilbert rank of their WARPED position, with a
        # tie-break on the cell's own Hilbert rank so the composite key is a
        # strict total order — hence ``indices`` is always a valid permutation,
        # and μ=0 (⇒ warped == identity) reproduces the fixed Hilbert order
        # exactly.
        rank = self.hilbert_rank.to(warped_flat.device)
        warped_rank = rank[warped_flat]  # [B, N]
        composite = warped_rank * n + rank.unsqueeze(0)
        indices = composite.argsort(dim=-1, stable=True)  # [B, N] int64
        return {"indices": indices, "mu": mu, "phi": phi}


__all__ = ["BeltramiSFCBlock"]
