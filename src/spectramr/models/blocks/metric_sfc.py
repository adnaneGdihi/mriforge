"""Foreground-restricted metric-aware space-filling-curve linearizer.

Generalizes :mod:`spectramr.models.blocks.topology_linearizer` (Hilbert /
Morton / snake / zigzag / radial — all full-grid, intensity-blind)
to a graph-based traversal that:

1. Restricts to the foreground subgraph defined by a brain mask.
2. Weights edges by ``||u - v|| * sqrt(1 + beta * (I(u) - I(v))^2)``
   so high-contrast boundaries are crossed less frequently.
3. Orders voxels by a Kruskal-MST + DFS-preorder traversal of that graph.

Honest characterisation (do not overstate). The MST + DFS-preorder is a
deterministic, connected traversal — **not** a bounded-approximation TSP tour:
the classic MST-doubling / Christofides guarantee requires a *complete* metric
graph (triangle inequality on shortcut edges), whereas this is a sparse
k-neighbour graph, so the DFS "jumps" back along the tree are non-edges of
unbounded voxel length. Empirically the ordering is mostly local (mean L1 step
≈ 1.7) but has occasional large jumps at backtracks and the inter-component
bridge; it is a heuristic content-aware curve, not a locality-bounded one.

The curve is also **fixed once**: the permutation is computed on the CPU at
``__init__`` (or, in :class:`GeoMambaUNet`, from the first batch's volume) and
stored as an ``nn.Buffer`` — it is data-adaptive to a single template, not
per-sample and not learned. (In ``g_mno`` it is built with zero intensity and an
all-ones mask, so there it reduces to a content-independent Euclidean MST.)

For the typical "canonical template space" use-case the same mask is reused
across many subjects after coregistration, so the MST cost is paid once per fit.
An optional on-disk cache keyed by ``(shape, mask_hash, beta)`` lets later
training runs reuse the result.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    connected_components,
    depth_first_order,
    minimum_spanning_tree,
)

logger = logging.getLogger(__name__)


def _foreground_neighbour_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """6 / 18 / 26-connected neighbour offsets in 3D."""
    if connectivity == 6:
        return [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
    if connectivity in (18, 26):
        offsets: list[tuple[int, int, int]] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    sq = dz * dz + dy * dy + dx * dx
                    if connectivity == 18 and sq > 2:
                        continue
                    offsets.append((dz, dy, dx))
        return offsets
    raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")


def _build_metric_graph(
    intensity_smoothed: np.ndarray,
    mask: np.ndarray,
    beta: float,
    connectivity: int,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the foreground graph with intensity-aware edge weights.

    Returns:
        (W, fg_indices) where W is an N×N sparse upper-triangular
        weight matrix (N = number of foreground voxels) and
        fg_indices maps row index → flat-spatial index in the volume.
    """
    fg_voxels = np.argwhere(mask > 0)
    n = fg_voxels.shape[0]
    if n == 0:
        raise ValueError("MetricSFCLinearizer: foreground mask is empty.")

    # Map (z, y, x) → row index for fast neighbour lookup.
    coord_to_row: dict[tuple[int, int, int], int] = {
        (int(coord[0]), int(coord[1]), int(coord[2])): i for i, coord in enumerate(fg_voxels)
    }

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for offset in _foreground_neighbour_offsets(connectivity):
        spatial_d = float(np.sqrt(sum(o * o for o in offset)))
        for i, (z, y, x) in enumerate(fg_voxels):
            nb = (int(z) + offset[0], int(y) + offset[1], int(x) + offset[2])
            j = coord_to_row.get(nb)
            if j is None or j <= i:
                # Only emit upper-triangular edges to avoid duplicates.
                continue
            di = float(intensity_smoothed[z, y, x] - intensity_smoothed[nb])
            w = spatial_d * float(np.sqrt(1.0 + beta * di * di))
            rows.append(i)
            cols.append(j)
            data.append(w)

    spatial_shape = intensity_smoothed.shape
    flat_idx = (
        fg_voxels[:, 0] * spatial_shape[1] * spatial_shape[2]
        + fg_voxels[:, 1] * spatial_shape[2]
        + fg_voxels[:, 2]
    ).astype(np.int64)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W, flat_idx


def _mst_dfs_permutation(W: csr_matrix) -> np.ndarray:
    """Kruskal MST → DFS-preorder traversal (a heuristic content-aware curve).

    Not a bounded-approximation TSP tour — see the module docstring for why the
    MST-doubling guarantee does not apply to this sparse graph.

    Handles a **disconnected** foreground graph by stitching per-component
    traversals. ``depth_first_order(mst, i_start=0)`` visits only node
    0's connected component, so on a mask with multiple components (two blobs,
    hemispheres, lesions, or the noisy auto-threshold mask GeoMamba-ULF builds)
    it silently dropped every voxel outside that component — pitfall #16. We
    instead traverse each connected component in a deterministic order (by its
    smallest node index, which — since rows follow ``np.argwhere`` / flat-index
    order — is the smallest flat spatial index) and concatenate, guaranteeing
    every foreground voxel is visited exactly once.
    """
    n = W.shape[0]
    n_comp, labels = connected_components(W, directed=False, connection="weak")
    if n_comp == 1:
        order, _ = depth_first_order(
            minimum_spanning_tree(W),
            i_start=0,
            directed=False,
            return_predecessors=True,
        )
        perm = order.astype(np.int64)
    else:
        parts: list[np.ndarray] = []
        # Components in deterministic order: by the smallest node they contain.
        for comp in sorted(range(n_comp), key=lambda c: int(np.argmax(labels == c))):
            nodes = np.flatnonzero(labels == comp)
            if nodes.size == 1:
                parts.append(nodes.astype(np.int64))
                continue
            sub = W[nodes][:, nodes]
            sub_mst = minimum_spanning_tree(sub)
            local, _ = depth_first_order(
                sub_mst, i_start=0, directed=False, return_predecessors=True
            )
            parts.append(nodes[local].astype(np.int64))
        perm = np.concatenate(parts)
    if perm.size != n:
        raise RuntimeError(
            f"_mst_dfs_permutation: visited {perm.size} of {n} foreground nodes "
            "— component stitching failed to cover the graph."
        )
    return perm


def _hash_inputs(
    spatial_shape: tuple[int, ...],
    mask: np.ndarray,
    intensity_smoothed: np.ndarray,
    beta: float,
    connectivity: int,
) -> str:
    h = hashlib.sha256()
    # Version token: bumping it invalidates every on-disk permutation cached by
    # a prior algorithm. v2 fixes the disconnected-mask voxel drop — the same
    # (shape, mask, intensity, beta, connectivity) now yields a different,
    # correct permutation, so stale v1 pickles must NOT be reused.
    h.update(b"metric_sfc_v2_stitch")
    h.update(repr(spatial_shape).encode())
    h.update(mask.astype(np.uint8).tobytes())
    h.update(intensity_smoothed.astype(np.float32).tobytes())
    h.update(repr(beta).encode())
    h.update(repr(connectivity).encode())
    return h.hexdigest()


class MetricSFCLinearizer(nn.Module):
    """Metric-aware foreground-restricted SFC permutation.

    Args:
        intensity_volume: ``(D, H, W)`` reference volume used to
            compute the per-edge intensity contrast (numpy or torch
            tensor). Typically a coregistered template image or
            the mean of the training cohort.
        mask: ``(D, H, W)`` binary foreground mask. Only voxels with
            ``mask > 0`` are visited.
        beta: intensity edge-weight coefficient. Higher beta
            penalizes contrast crossings more strongly.
        smoothing_sigma: Gaussian sigma (in voxels) for the
            pre-denoise proxy used by the metric.
        connectivity: 6, 18, or 26.
        cache_dir: optional directory for pickled (shape, hash) →
            permutation cache. ``None`` disables caching.
    """

    def __init__(
        self,
        intensity_volume: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        beta: float = 10.0,
        smoothing_sigma: float = 1.0,
        connectivity: Literal[6, 18, 26] = 26,
        cache_dir: str | Path | None = "cache/metric_sfc",
    ) -> None:
        super().__init__()
        intensity_np = (
            intensity_volume.detach().cpu().numpy()
            if isinstance(intensity_volume, torch.Tensor)
            else np.asarray(intensity_volume)
        )
        mask_np = (
            mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        )
        if intensity_np.ndim != 3 or mask_np.ndim != 3:
            raise ValueError(
                "intensity_volume and mask must both be 3D (D, H, W); got "
                f"{intensity_np.shape} and {mask_np.shape}"
            )
        if intensity_np.shape != mask_np.shape:
            raise ValueError(
                "intensity_volume and mask shapes must match "
                f"({intensity_np.shape} vs {mask_np.shape})"
            )

        self.spatial_shape = tuple(int(s) for s in intensity_np.shape)
        self.beta = float(beta)
        self.smoothing_sigma = float(smoothing_sigma)
        self.connectivity = int(connectivity)

        smoothed = (
            gaussian_filter(intensity_np.astype(np.float32), sigma=smoothing_sigma)
            if smoothing_sigma > 0
            else intensity_np.astype(np.float32)
        )

        permutation_local, flat_idx = self._compute_or_load(smoothed, mask_np, cache_dir)

        # `permutation_local` is row-order in the foreground subgraph;
        # `flat_idx` maps row → flat-spatial index. Compose to get the
        # permutation in flat-spatial coordinates.
        forward_idx = flat_idx[permutation_local]
        # Inverse: spatial_idx → position in sequence (or -1 for bg).
        n_voxels = int(np.prod(self.spatial_shape))
        inverse_idx = -np.ones(n_voxels, dtype=np.int64)
        inverse_idx[forward_idx] = np.arange(len(forward_idx), dtype=np.int64)

        self.register_buffer("forward_idx", torch.from_numpy(forward_idx))
        self.register_buffer("inverse_idx", torch.from_numpy(inverse_idx))
        self.register_buffer("foreground_flat", torch.from_numpy(flat_idx))
        self.n_foreground = len(forward_idx)

        logger.info(
            "[MetricSFCLinearizer] shape=%s, foreground=%d/%d (%.1f%%), beta=%.2f",
            self.spatial_shape,
            self.n_foreground,
            n_voxels,
            100.0 * self.n_foreground / max(n_voxels, 1),
            self.beta,
        )

    # ------------------------------------------------------------------
    # Permutation construction (with on-disk cache)
    # ------------------------------------------------------------------

    def _compute_or_load(
        self,
        smoothed: np.ndarray,
        mask: np.ndarray,
        cache_dir: str | Path | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_path: Path | None = None
        if cache_dir is not None:
            key = _hash_inputs(self.spatial_shape, mask, smoothed, self.beta, self.connectivity)
            cache_path = Path(cache_dir) / f"sfc_{key}.pkl"
            if cache_path.exists():
                with cache_path.open("rb") as f:
                    record = pickle.load(f)
                logger.info("[MetricSFCLinearizer] loaded cached permutation: %s", cache_path)
                return record["permutation"], record["flat_idx"]

        W, flat_idx = _build_metric_graph(smoothed, mask, self.beta, self.connectivity)
        permutation = _mst_dfs_permutation(W)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump({"permutation": permutation, "flat_idx": flat_idx}, f)
            logger.info("[MetricSFCLinearizer] cached permutation to %s", cache_path)

        return permutation, flat_idx

    # ------------------------------------------------------------------
    # Forward / reverse
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute (B, C, D, H, W) → (B, C, N_F) along the SFC."""
        if x.shape[-3:] != self.spatial_shape:
            raise ValueError(
                "Input spatial shape mismatch: expected "
                f"{self.spatial_shape}, got {tuple(x.shape[-3:])}"
            )
        B, C = x.shape[:2]
        flat = x.reshape(B, C, -1)
        return flat[:, :, self.forward_idx]

    def reverse(
        self, sequence: torch.Tensor, background: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Re-embed (B, C, N_F) into (B, C, D, H, W).

        Background voxels are filled with ``background`` if provided
        (must broadcast to the spatial shape), else with zeros.
        """
        B, C, N = sequence.shape
        if self.n_foreground != N:
            raise ValueError(f"sequence length {N} != foreground voxel count {self.n_foreground}")
        D, H, W = self.spatial_shape
        n_voxels = D * H * W
        device = sequence.device
        dtype = sequence.dtype
        if background is None:
            out = torch.zeros((B, C, n_voxels), device=device, dtype=dtype)
        else:
            out = background.reshape(B, C, n_voxels).clone()
        out[:, :, self.forward_idx] = sequence
        return out.reshape(B, C, D, H, W)

    def extra_repr(self) -> str:
        return (
            f"shape={self.spatial_shape}, n_foreground={self.n_foreground}, "
            f"beta={self.beta}, connectivity={self.connectivity}"
        )


__all__ = ["MetricSFCLinearizer"]
