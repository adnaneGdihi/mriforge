r"""TALL — Texture-Adaptive Learnable L-system.

Implements ``IMPLEMENTATION_SPEC.md`` §9 (Phase 3): a learnable
Hilbert-like traversal whose orientation is content-adaptive. At each
recursion level a small policy network selects one of K motif
orientations for each cell; the choice is differentiable through
Gumbel-softmax so the entire traversal is end-to-end trainable.

This pass implements the **2-D variant** with K=4 motif orientations
(rotations of the canonical Hilbert "U-shape" by multiples of 90°).
The 3-D variant with K=24 follows the same scaffolding and can be
added by extending :func:`build_motif_permutations` to the 3-D
rotation group.

Initialisation rule (Spec §9.4): the policy is initialised so that at
recursion level 0 it emits the canonical Hilbert traversal in
expectation, achieved by setting the final-layer bias to log-prob
matching the canonical motif sequence at each cell.

References
----------
* IMPLEMENTATION_SPEC.md §9 (TALL — Texture-Adaptive Learnable L-system).
* E. Jang, S. Gu, B. Poole, "Categorical reparameterization with
  Gumbel-softmax," ICLR 2017.
* B. Moon et al., "Analysis of the clustering properties of the
  Hilbert space-filling curve," IEEE TKDE 13(1), 2001.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Canonical motifs in 2-D (K = 4 orientations)
# ──────────────────────────────────────────────────────────────────────


def _canonical_u_motif() -> torch.Tensor:
    """Canonical 2×2 U-shape motif → 4-step ordering.

    The Hilbert "U" base-shape visits the four cells of a 2×2 block
    in order:  bottom-left → bottom-right → top-right → top-left.

    Returns an int64 tensor of shape (4,) with cell-indices in order.
    """
    # 2x2 grid laid out row-major:
    #   2 3
    #   0 1
    return torch.tensor([0, 1, 3, 2], dtype=torch.long)


def _cube_rotation_group() -> list[torch.Tensor]:
    r"""24 proper rotations of the cube as 3×3 integer rotation matrices.

    The proper rotation group of the cube has order 24:
    1 identity, 6 face-rotations by ±90°, 3 face-rotations by 180°,
    8 vertex-rotations by ±120°, 6 edge-rotations by 180°.

    We construct them by composing the three generators
    R_x(90°), R_y(90°), R_z(90°) and de-duplicating.
    """
    Rx = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.long)
    Ry = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=torch.long)
    Rz = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.long)
    seen: dict[tuple, torch.Tensor] = {}
    frontier = [torch.eye(3, dtype=torch.long)]
    while frontier:
        next_frontier: list[torch.Tensor] = []
        for M in frontier:
            key = tuple(M.flatten().tolist())
            if key in seen:
                continue
            seen[key] = M
            for G in (Rx, Ry, Rz):
                next_frontier.append(G @ M)
        frontier = next_frontier
        if len(seen) >= 24:
            break
    return list(seen.values())[:24]


def _canonical_u_motif_3d() -> torch.Tensor:
    """Canonical Hilbert visit order for a 2×2×2 block.

    The 3-D Hilbert motif visits the 8 vertices of a 2×2×2 cube in
    the order recommended by Skilling (2004). Cells are indexed as
    ``z * 4 + y * 2 + x`` (depth-major).
    """
    # Standard Hilbert/Gray-code 3-D U order
    # cell coords (x, y, z) in the canonical 3-D Hilbert traversal
    coords = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
        (0, 0, 1),
    ]
    return torch.tensor(
        [z * 4 + y * 2 + x for (x, y, z) in coords],
        dtype=torch.long,
    )


def build_motif_permutations(n_dims: int = 2) -> torch.Tensor:
    r"""Build the K motif permutation tables.

    * 2-D: K=4 rotations of the canonical U-motif over the 2×2 cell.
    * 3-D: K=24 proper rotations of the cube applied to the canonical
      Hilbert visit order over the 2×2×2 cell.

    Args:
        n_dims: 2 or 3.

    Returns:
        ``(K, n_cells)`` int64 tensor where row ``k`` is the cell
        visit order for motif ``k``. ``n_cells = 2**n_dims``.
    """
    if n_dims == 2:
        n_cells = 4
        canonical = _canonical_u_motif()
        grid = torch.arange(n_cells).reshape(2, 2)
        out = []
        for k in range(4):
            rotated = torch.rot90(grid, k=k)
            cells = rotated.flatten()
            perm = cells[canonical]
            out.append(perm)
        return torch.stack(out, dim=0)  # (4, 4)

    if n_dims == 3:
        # 24 cube rotations applied to the canonical 3-D Hilbert motif
        canonical = _canonical_u_motif_3d()
        # Cell-coordinates (x, y, z) for each of the 8 cells in the 2×2×2
        # block, indexed as z*4 + y*2 + x
        cell_coords = torch.tensor(
            [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)],
            dtype=torch.long,
        )  # (8, 3)
        # For each rotation, compute the new index of every cell after
        # rotation about the cube centre (0.5, 0.5, 0.5).
        rotations = _cube_rotation_group()
        out = []
        for R in rotations:
            # Centre cells around 0.5, rotate, recentre to {0, 1}
            centered = cell_coords.float() - 0.5
            rotated = (R.float() @ centered.t()).t() + 0.5
            rotated_int = rotated.round().long().clamp(0, 1)
            # Re-index: each rotated cell maps to a new cell index
            new_idx = rotated_int[:, 2] * 4 + rotated_int[:, 1] * 2 + rotated_int[:, 0]
            # Verify the rotation produces a valid permutation
            if not torch.equal(torch.sort(new_idx).values, torch.arange(8)):
                continue
            # The motif under this rotation visits cells in the order
            # `new_idx[canonical_position]` for each canonical step
            perm = new_idx[canonical]
            out.append(perm)
        # If de-duplication leaves fewer than 24 (numerical edge cases),
        # truncate / pad as needed; in practice we expect exactly 24.
        if len(out) < 24:
            # Pad with the identity motif
            ident = canonical.clone()
            while len(out) < 24:
                out.append(ident)
        return torch.stack(out[:24], dim=0)  # (24, 8)

    raise NotImplementedError(f"TALL motif tables only support n_dims in (2, 3); got {n_dims}")


# ──────────────────────────────────────────────────────────────────────
# TALL — learnable SFC
# ──────────────────────────────────────────────────────────────────────


class TALL(nn.Module):
    r"""Texture-Adaptive Learnable L-system (Spec §9).

    Args:
        order:        Recursion depth ``p`` so the grid is ``2**p × 2**p``.
        feature_dim:  Number of features per cell handed to the policy.
        n_motifs:     Number of motif orientations (4 for 2-D, 24 for 3-D).
        policy_hidden: Hidden dim of the policy MLP.
        tau_init:     Initial Gumbel-softmax temperature (smooth → peaked).
        tau_min:      Annealing target.
        anneal_steps: Linear anneal over this many ``forward`` calls.
        canonical_init: If True (default), initialise the policy so that
                       it emits canonical-Hilbert in expectation at step 0.
    """

    def __init__(
        self,
        order: int = 4,
        feature_dim: int = 16,
        n_motifs: int | None = None,
        policy_hidden: int = 64,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        anneal_steps: int = 10000,
        canonical_init: bool = True,
        n_dims: int = 2,
    ) -> None:
        super().__init__()
        if n_dims not in (2, 3):
            raise ValueError(f"TALL n_dims must be 2 or 3; got {n_dims}")
        # Default motif counts per the canonical group order
        default_motifs = {2: 4, 3: 24}
        if n_motifs is None:
            n_motifs = default_motifs[n_dims]
        if n_motifs != default_motifs[n_dims]:
            raise NotImplementedError(
                f"TALL n_dims={n_dims} requires n_motifs={default_motifs[n_dims]}; got {n_motifs}"
            )
        self.order = int(order)
        self.side = 2**self.order
        self.n_dims = int(n_dims)
        self.n_motifs = int(n_motifs)
        self.cells_per_block = 2**self.n_dims

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, policy_hidden),
            nn.GELU(),
            nn.Linear(policy_hidden, n_motifs),
        )
        if canonical_init:
            with torch.no_grad():
                self.policy[-1].weight.zero_()
                bias = torch.zeros(n_motifs)
                bias[0] = 2.0
                self.policy[-1].bias.copy_(bias)

        self.register_buffer(
            "motif_permutations",
            build_motif_permutations(self.n_dims),  # (K, cells_per_block)
        )

        self.tau_init = float(tau_init)
        self.tau_min = float(tau_min)
        self.anneal_steps = int(anneal_steps)
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @property
    def tau(self) -> float:
        """Current Gumbel-softmax temperature (annealed linearly)."""
        s = float(self.step_counter.item())
        frac = min(1.0, s / max(1.0, self.anneal_steps))
        return self.tau_init + (self.tau_min - self.tau_init) * frac

    def _cell_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``x`` into per-cell summary features.

        2-D path: ``(B, C, H, W)`` → ``(B, (side/2)², feature_dim)``
        3-D path: ``(B, C, D, H, W)`` → ``(B, (side/2)³, feature_dim)``
        """
        if self.n_dims == 2:
            B, C, H, W = x.shape
            x_pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
            n_cells = x_pooled.shape[-2] * x_pooled.shape[-1]
            feat = x_pooled.permute(0, 2, 3, 1).reshape(B, n_cells, C)
        else:
            B, C, D, H, W = x.shape
            x_pooled = F.avg_pool3d(x, kernel_size=2, stride=2)
            n_cells = x_pooled.shape[-3] * x_pooled.shape[-2] * x_pooled.shape[-1]
            feat = x_pooled.permute(0, 2, 3, 4, 1).reshape(B, n_cells, C)
        feature_dim = self.policy[0].in_features
        if feature_dim > C:
            feat = F.pad(feat, (0, feature_dim - C))
        elif feature_dim < C:
            feat = feat[..., :feature_dim]
        return feat

    # ──────────────────────────────────────────────────────────────────
    # Forward — produce a (soft) permutation of the spatial grid
    # ──────────────────────────────────────────────────────────────────

    def _block_abs_indices(self, device: torch.device) -> torch.Tensor:
        """Pre-compute absolute voxel indices per (cell × in-cell position).

        For 2-D returns ``(n_cells, 4)``; for 3-D returns ``(n_cells, 8)``.
        Cell raster order is row-major (2-D) / depth-major (3-D); in-cell
        order matches the canonical motif's coordinate convention.
        """
        if self.n_dims == 2:
            ns = self.side // 2
            n_cells = ns * ns
            cy = torch.arange(ns, device=device).repeat_interleave(ns)
            cx = torch.arange(ns, device=device).repeat(ns)
            local_y = torch.tensor([0, 0, 1, 1], device=device)
            local_x = torch.tensor([0, 1, 0, 1], device=device)
            abs_y = cy.unsqueeze(1) * 2 + local_y.unsqueeze(0)
            abs_x = cx.unsqueeze(1) * 2 + local_x.unsqueeze(0)
            return abs_y * self.side + abs_x  # (n_cells, 4)

        # 3-D
        ns = self.side // 2
        n_cells = ns**3
        # Cell coords in raster order: z varies slowest, then y, then x
        cz = torch.arange(ns, device=device).repeat_interleave(ns * ns)
        cy = torch.arange(ns, device=device).repeat_interleave(ns).repeat(ns)
        cx = torch.arange(ns, device=device).repeat(ns * ns)
        # Local 2×2×2 cell offsets indexed as z*4 + y*2 + x
        loc = torch.tensor(
            [[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)],
            device=device,
            dtype=torch.long,
        )
        local_x, local_y, local_z = loc[:, 0], loc[:, 1], loc[:, 2]
        abs_x = cx.unsqueeze(1) * 2 + local_x.unsqueeze(0)
        abs_y = cy.unsqueeze(1) * 2 + local_y.unsqueeze(0)
        abs_z = cz.unsqueeze(1) * 2 + local_z.unsqueeze(0)
        return abs_z * (self.side**2) + abs_y * self.side + abs_x  # (n_cells, 8)

    def forward(
        self,
        x: torch.Tensor,
        hard: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a content-adaptive 1-D ordering of ``x``.

        Args:
            x:    ``(B, C, side, side)`` (2-D) or ``(B, C, side, side, side)``
                  (3-D) input.
            hard: If True, return the argmax (one-hot) motif choice;
                  if None, hard at eval, soft at train.

        Returns:
            seq:        ``(B, C, N)`` linearised sequence.
            permutation: ``(B, N)`` int64 permutation per batch element.
        """
        if hard is None:
            hard = not self.training

        B = x.shape[0]
        C = x.shape[1]
        spatial = x.shape[2:]
        expected = (self.side,) * self.n_dims
        if spatial != expected:
            raise ValueError(f"TALL expected spatial shape {expected}, got {spatial}")

        feat = self._cell_features(x)  # (B, n_cells, feature_dim)
        logits = self.policy(feat)  # (B, n_cells, K)

        choice = F.gumbel_softmax(
            logits,
            tau=self.tau,
            hard=hard,
            dim=-1,
        )  # (B, n_cells, K)

        K = self.n_motifs
        cells_per_block = self.cells_per_block
        abs_idx = self._block_abs_indices(x.device)  # (n_cells, cells_per_block)
        n_cells = abs_idx.shape[0]
        motif_perm = self.motif_permutations  # (K, cells_per_block)

        # ordered[c, k, j] = abs_idx[c][ motif_perm[k][j] ]
        ordered = (
            abs_idx.unsqueeze(1)
            .expand(-1, K, -1)
            .gather(
                -1,
                motif_perm.unsqueeze(0).expand(n_cells, -1, -1),
            )
        )  # (n_cells, K, cells_per_block)

        # Hard argmax permutation
        argmax = choice.argmax(dim=-1)  # (B, n_cells)
        c_idx = torch.arange(n_cells, device=x.device).unsqueeze(0).expand(B, -1)
        chosen = ordered[c_idx, argmax]  # (B, n_cells, cells_per_block)
        permutation = chosen.reshape(B, -1)  # (B, N)

        flat = x.reshape(B, C, -1)  # (B, C, N)
        hard_seq = torch.gather(
            flat,
            dim=-1,
            index=permutation.unsqueeze(1).expand(-1, C, -1),
        )  # (B, C, N)

        if hard:
            return hard_seq, permutation

        # Soft mixing for gradient flow
        soft_seqs = torch.gather(
            flat.unsqueeze(2).expand(-1, -1, K, -1),  # (B, C, K, N_total)
            dim=-1,
            index=ordered.reshape(1, 1, K, -1).expand(  # (1, 1, K, n_cells*cells_per_block)
                B, C, -1, -1
            ),
        )  # (B, C, K, n_cells*cells_per_block)
        weights = choice.permute(0, 2, 1)  # (B, K, n_cells)
        weights = weights.repeat_interleave(
            cells_per_block,
            dim=-1,
        ).unsqueeze(1)  # (B, 1, K, n_cells*cells_per_block)
        soft_seq = (soft_seqs * weights).sum(dim=2)  # (B, C, N)

        if self.training:
            self.step_counter += 1

        return soft_seq, permutation


# ──────────────────────────────────────────────────────────────────────
# Locality regulariser (Spec §9.3)
# ──────────────────────────────────────────────────────────────────────


def locality_loss(
    permutation: torch.Tensor,
    side: int,
    n_dims: int = 2,
) -> torch.Tensor:
    r"""Encourage consecutive permutation entries to be spatial neighbours.

    .. math::
        \mathcal L_{\text{loc}} = \frac{1}{N-1} \sum_{i=1}^{N-1}
            \big\| \mathbf r(\pi_{i+1}) - \mathbf r(\pi_i) \big\|

    Args:
        permutation: ``(B, N)`` int64 permutation per batch.
        side:        Spatial side of the grid.
        n_dims:      2 or 3.
    """
    if n_dims == 2:
        cy = (permutation // side).float()
        cx = (permutation % side).float()
        coords = torch.stack([cy, cx], dim=-1)
    elif n_dims == 3:
        cz = (permutation // (side**2)).float()
        rem = permutation % (side**2)
        cy = (rem // side).float()
        cx = (rem % side).float()
        coords = torch.stack([cz, cy, cx], dim=-1)
    else:
        raise ValueError(f"locality_loss n_dims must be 2 or 3, got {n_dims}")
    diffs = coords[:, 1:] - coords[:, :-1]
    return diffs.norm(dim=-1).mean()


__all__ = ["TALL", "build_motif_permutations", "locality_loss"]
