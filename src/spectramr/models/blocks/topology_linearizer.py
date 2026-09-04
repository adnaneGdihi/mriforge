"""Unified Image Topology Linearizer for Vision State-Space Models.

Provides multiple space-filling curve strategies to map 2D/3D grids into
locality-preserving 1D sequences. All permutation indices are pre-computed
**once** on the CPU at ``__init__`` and stored as ``nn.Buffer`` for O(1)
GPU execution during training.

Supported modes:

- ``hilbert_2d``: Hilbert fractal curve (platinum standard for 2D locality).
  Requires square, power-of-2 dimensions.
- ``hilbert_3d``: 3D Hilbert curve for volumetric data.  Preserves 3D
  spatial locality through recursive octant subdivision.  Requires cubic,
  power-of-2 dimensions.
- ``morton_2d``: Z-order curve via bit-interleaving (Quadtree ordering).
  Works on any grid size.
- ``morton_3d``: 3D Morton code (Octree ordering for volumetric data).
- ``snake_2d``: Boustrophedon (serpentine) scan eliminating carriage-return
  tears at row boundaries.
- ``zigzag_2d``: Diagonal sweep from top-left to bottom-right, ideal for
  frequency-domain ordering (low→high).
- ``radial_inside_out``: Centrifugal concentric shell sweep from FOV center
  outward.  Prioritizes deep-brain anatomy over background air.
- ``radial_outside_in``: Centripetal concentric shell sweep from FOV edge
  inward.  Acts as a degradation calibrator for motion correction.

Reference:
    Zhu et al. (2024). "Vision Mamba." arXiv:2401.09417.
    Liu et al. (2024). "VMamba." arXiv:2401.10166.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index generators (CPU-only, called once at __init__)
# ---------------------------------------------------------------------------


def _hilbert_2d_matrix(n: int) -> np.ndarray:
    """Recursively generate a 2D Hilbert curve ordering matrix.

    Args:
        n: Side length (must be power of 2).

    Returns:
        n×n array where entry (r, c) is the Hilbert visit index.
    """
    if n == 1:
        return np.zeros((1, 1), dtype=np.int64)
    half = n // 2
    prev = _hilbert_2d_matrix(half)
    offset = half * half

    # Four quadrants: TL=transposed, TR=offset, BR=2*offset, BL=rot180+transposed
    tl = prev.T
    tr = prev + offset
    bl = np.flipud(np.fliplr(prev)).T + 3 * offset
    br = prev + 2 * offset

    return np.vstack((np.hstack((tl, tr)), np.hstack((bl, br))))


def _hilbert_2d_indices(h: int, w: int) -> np.ndarray:
    """Generate Hilbert-curve permutation for a 2D grid.

    Args:
        h: Grid height (must equal w, power of 2).
        w: Grid width (must equal h, power of 2).

    Returns:
        1D array of length h*w: permutation from raster to Hilbert order.
    """
    assert h == w and (h & (h - 1)) == 0, (
        f"Hilbert 2D requires square power-of-2 dims, got ({h}, {w})"
    )
    matrix = _hilbert_2d_matrix(h)
    return np.argsort(matrix.flatten())


def _rect_hilbert_2d_indices(h: int, w: int) -> np.ndarray:
    """Hilbert permutation for an ARBITRARY 2-D grid (any ``h``, ``w``).

    Generalises :func:`_hilbert_2d_indices` (square power-of-2 only) to any
    rectangle by laying the canonical Hilbert curve over the smallest
    enclosing square of side ``n = next_pow2(max(h, w))`` and keeping only the
    cells inside the ``h x w`` window, in curve-visit order. The result is a
    valid permutation of ``range(h*w)`` whose locality still follows the
    Hilbert ordering (consecutive tokens stay spatially close).

    For a square power-of-2 grid the enclosing square IS the grid, so this
    reduces **byte-identically** to :func:`_hilbert_2d_indices`.

    Used by :class:`~spectramr.models.generators.cs_mno_operator.CSMNOOperator`
    on its dynamic (validation) shapes so the operator is genuinely
    resolution-agnostic instead of crashing on a non-square / non-power-of-2
    full-resolution volume (F37 follow-up, 2026-06).
    """
    n = 1 << (max(h, w) - 1).bit_length()  # smallest power of 2 >= max(h, w)
    matrix = _hilbert_2d_matrix(n)
    sub = matrix[:h, :w]
    return np.argsort(sub.flatten())


def _skilling_hilbert_matrix(n: int, ndim: int) -> np.ndarray:
    """Locality-preserving Hilbert visit-index matrix via Skilling's algorithm.

    Implements the transpose form of the Hilbert curve (J. Skilling,
    "Programming the Hilbert curve," AIP Conf. Proc. 707, 2004), vectorised over
    all ``n**ndim`` distances with numpy bit operations. Returns an
    ``(n,)*ndim`` int64 array whose entry at a coordinate is that voxel's index
    along the curve — i.e. ``mat[coord] == distance``.

    Unlike a naive octant recursion (which is easy to get *bijective but not
    locality-preserving*), Skilling's construction guarantees that consecutive
    distances map to L1-adjacent coordinates (max step 1). ``n`` must be a power
    of two. For ``ndim == 2`` this agrees in locality with
    :func:`_hilbert_2d_matrix` (a possibly-different orientation of the same
    curve); the 2-D production path keeps its own recursion, so this is used for
    ``ndim == 3``.

    Complexity is ``O(p)`` numpy passes over ``n**ndim`` elements (``p = log2
    n``), acceptable for the one-time construction cached in the linearizer's
    buffers.
    """
    p = int(round(np.log2(n)))
    if (1 << p) != n:
        raise ValueError(f"Skilling Hilbert requires a power-of-two side, got {n}")
    total = n**ndim
    h = np.arange(total, dtype=np.uint64)

    # De-interleave the distance into the transpose X[ndim, total]: bit ``j`` of
    # axis ``i`` is bit ``j*ndim + (ndim-1-i)`` of the distance. (This exact
    # interleave — dim index reversed within each level — is the one that yields
    # max L1 step 1; the obvious ``+i`` ordering does not.)
    X = np.zeros((ndim, total), dtype=np.uint64)
    for j in range(p):
        for i in range(ndim):
            src = np.uint64(j * ndim + (ndim - 1 - i))
            X[i] |= ((h >> src) & np.uint64(1)) << np.uint64(j)

    # Gray-decode: H ^ (H >> 1) across the transpose.
    t = X[ndim - 1] >> np.uint64(1)
    for i in range(ndim - 1, 0, -1):
        X[i] ^= X[i - 1]
    X[0] ^= t

    # Undo excess work (Skilling). Note axis 0 aliases itself, so the i==0
    # iteration reduces to the invert-only branch (the exchange is a no-op).
    top = np.uint64(1) << np.uint64(p)
    q = np.uint64(2)
    while q != top:
        pmask = q - np.uint64(1)
        for i in range(ndim - 1, -1, -1):
            has_bit = (X[i] & q) != 0
            if i == 0:
                X[0] = np.where(has_bit, X[0] ^ pmask, X[0])
            else:
                swap = (X[0] ^ X[i]) & pmask
                x0 = np.where(has_bit, X[0] ^ pmask, X[0] ^ swap)
                xi = np.where(has_bit, X[i], X[i] ^ swap)
                X[0], X[i] = x0, xi
        q <<= np.uint64(1)

    coords = tuple(X[d].astype(np.int64) for d in range(ndim))
    mat = np.zeros((n,) * ndim, dtype=np.int64)
    mat[coords] = h.astype(np.int64)
    return mat


def _hilbert_3d_matrix(n: int) -> np.ndarray:
    """3D Hilbert curve ordering matrix (locality-preserving).

    Delegates to :func:`_skilling_hilbert_matrix`. The earlier recursive octant
    construction produced a valid permutation whose consecutive voxels could be
    up to L1 distance 12 apart (n=8), silently defeating the whole point of a
    Hilbert scan (see ``TestHilbert3DLocality``).

    Args:
        n: Side length of the cube (must be power of 2).

    Returns:
        n×n×n array where entry (d, r, c) is the Hilbert visit index.
    """
    if n == 1:
        return np.zeros((1, 1, 1), dtype=np.int64)
    return _skilling_hilbert_matrix(n, 3)


def _hilbert_3d_indices(d: int, h: int, w: int) -> np.ndarray:
    """Generate Hilbert-curve permutation for a 3D grid.

    Args:
        d: Grid depth (must equal h and w, power of 2).
        h: Grid height (must equal d and w, power of 2).
        w: Grid width (must equal d and h, power of 2).

    Returns:
        1D array of length d*h*w: permutation from raster to Hilbert order.
    """
    assert d == h == w and (d & (d - 1)) == 0, (
        f"Hilbert 3D requires cubic power-of-2 dims, got ({d}, {h}, {w})"
    )
    matrix = _hilbert_3d_matrix(d)
    return np.argsort(matrix.flatten())


def _rect_hilbert_3d_indices(d: int, h: int, w: int) -> np.ndarray:
    """Hilbert permutation for an ARBITRARY 3-D grid (any ``d``, ``h``, ``w``).

    The 3-D analogue of :func:`_rect_hilbert_2d_indices`: lay the canonical
    cubic Hilbert curve over the smallest enclosing cube of side
    ``n = next_pow2(max(d, h, w))`` and keep only the in-bounds ``d x h x w``
    voxels in curve-visit order. Reduces byte-identically to
    :func:`_hilbert_3d_indices` on a cubic power-of-2 grid.
    """
    n = 1 << (max(d, h, w) - 1).bit_length()  # smallest power of 2 >= max dim
    matrix = _hilbert_3d_matrix(n)
    sub = matrix[:d, :h, :w]
    return np.argsort(sub.flatten())


def _morton_2d_indices(h: int, w: int) -> np.ndarray:
    """Generate Morton Z-order permutation for a 2D grid."""
    from spectramr.models.blocks.morton_stem import compute_morton_2d

    fwd, _ = compute_morton_2d(h, w)
    return fwd.numpy()


def _morton_3d_indices(d: int, h: int, w: int) -> np.ndarray:
    """Generate Morton Z-order permutation for a 3D grid."""
    from spectramr.models.blocks.morton_stem import compute_morton_3d

    fwd, _ = compute_morton_3d(d, h, w)
    return fwd.numpy()


def _snake_2d_indices(h: int, w: int) -> np.ndarray:
    """Generate boustrophedon (snake) permutation for a 2D grid.

    Even rows read left→right, odd rows read right→left.
    Eliminates the carriage-return spatial tear at row boundaries.
    """
    grid = np.arange(h * w).reshape(h, w)
    grid[1::2, :] = grid[1::2, ::-1]
    return grid.flatten()


def _zigzag_2d_indices(h: int, w: int) -> np.ndarray:
    """Generate diagonal zig-zag permutation for a 2D grid.

    Sweeps from top-left to bottom-right along anti-diagonals.
    Ideal for ordering frequency-domain patches (low → high).
    """
    result = np.zeros((h, w), dtype=np.int64)
    idx = 0

    for s in range(h + w - 1):
        if s % 2 == 0:
            # Sweep up-right
            r = min(s, h - 1)
            c = s - r
            while r >= 0 and c < w:
                result[r, c] = idx
                idx += 1
                r -= 1
                c += 1
        else:
            # Sweep down-left
            c = min(s, w - 1)
            r = s - c
            while c >= 0 and r < h:
                result[r, c] = idx
                idx += 1
                c -= 1
                r += 1

    return np.argsort(result.flatten())


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------


def _radial_inside_out(h: int, w: int) -> np.ndarray:
    """Centrifugal (center→edge) concentric pixel permutation."""
    from spectramr.models.blocks.radial_linearizer import radial_inside_out_indices

    return radial_inside_out_indices(h, w)


def _radial_outside_in(h: int, w: int) -> np.ndarray:
    """Centripetal (edge→center) concentric pixel permutation."""
    from spectramr.models.blocks.radial_linearizer import radial_outside_in_indices

    return radial_outside_in_indices(h, w)


def _raster_indices(*dims: int) -> np.ndarray:
    """Identity permutation: tokens visited in row-major (C-contiguous) order.

    Used as the LMO-baseline scan order in CS-MNO ablations.
    """
    n = int(np.prod(dims))
    return np.arange(n, dtype=np.int64)


_MODE_GENERATORS = {
    "hilbert_2d": lambda shape: _hilbert_2d_indices(*shape[:2]),
    "hilbert_3d": lambda shape: _hilbert_3d_indices(*shape[:3]),
    # Resolution-agnostic Hilbert: identical to the strict variants on a
    # square/cubic power-of-2 grid, but valid on any rectangle/cuboid.
    "hilbert_2d_rect": lambda shape: _rect_hilbert_2d_indices(*shape[:2]),
    "hilbert_3d_rect": lambda shape: _rect_hilbert_3d_indices(*shape[:3]),
    "morton_2d": lambda shape: _morton_2d_indices(*shape[:2]),
    "morton_3d": lambda shape: _morton_3d_indices(*shape[:3]),
    "snake_2d": lambda shape: _snake_2d_indices(*shape[:2]),
    "zigzag_2d": lambda shape: _zigzag_2d_indices(*shape[:2]),
    "radial_inside_out": lambda shape: _radial_inside_out(*shape[:2]),
    "radial_outside_in": lambda shape: _radial_outside_in(*shape[:2]),
    "raster_1d": lambda shape: _raster_indices(*shape[:1]),
    "raster_2d": lambda shape: _raster_indices(*shape[:2]),
    "raster_3d": lambda shape: _raster_indices(*shape[:3]),
}


# Shared SSOT for the "strict Hilbert -> resolution-agnostic Hilbert" remap.
# The strict ``hilbert_2d`` / ``hilbert_3d`` builders assert square/cubic
# power-of-two extents; their ``*_rect`` siblings crop the canonical curve out
# of the smallest enclosing square/cube, reducing byte-identically on strict
# shapes but staying valid on any extent. Call sites that must accept dynamic
# (e.g. full-resolution validation) shapes route through
# :func:`resolution_agnostic_mode`; call sites that deliberately keep the strict
# fail-loud contract (e.g. ``MambaLayer2D``) do NOT.
RESOLUTION_AGNOSTIC_SCAN: dict[str, str] = {
    "hilbert_2d": "hilbert_2d_rect",
    "hilbert_3d": "hilbert_3d_rect",
}


def resolution_agnostic_mode(mode: str) -> str:
    """Map a strict Hilbert scan mode to its resolution-agnostic ``*_rect``
    sibling; every other mode (already shape-agnostic — morton/snake/raster/
    radial, or an explicit ``*_rect``) passes through unchanged."""
    return RESOLUTION_AGNOSTIC_SCAN.get(mode, mode)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ImageTopologyLinearizer(nn.Module):
    """Configurable topology-preserving linearizer for 2D/3D grids.

    Pre-computes the permutation at init (CPU) and applies it as a
    zero-cost index gather on the GPU during forward pass.

    Args:
        shape: Spatial dimensions — ``(H, W)`` for 2D or ``(D, H, W)`` for 3D.
        mode: Linearization strategy. One of:
            ``"hilbert_2d"``, ``"morton_2d"``, ``"morton_3d"``,
            ``"snake_2d"``, ``"zigzag_2d"``.

    Example::

        lin = ImageTopologyLinearizer((256, 256), mode="hilbert_2d")
        seq = lin(img)        # [B, C, H*W] in Hilbert order
        img = lin.reverse(seq)  # [B, C, H, W] restored
    """

    def __init__(self, shape: tuple[int, ...], mode: str = "hilbert_2d") -> None:
        super().__init__()
        if mode not in _MODE_GENERATORS:
            raise ValueError(
                f"Unknown linearization mode '{mode}'. Available: {list(_MODE_GENERATORS.keys())}"
            )

        self.shape = shape
        self.mode = mode

        # Pre-compute permutation ONCE on CPU
        indices = _MODE_GENERATORS[mode](shape)
        self.register_buffer("forward_idx", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("inverse_idx", torch.argsort(self.forward_idx))

        logger.info(
            "[ImageTopologyLinearizer] mode=%s, shape=%s, seq_len=%d",
            mode,
            shape,
            len(indices),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linearize spatial dims into topology-preserving 1D sequence.

        Args:
            x: Input tensor ``[B, C, H, W]`` or ``[B, C, D, H, W]``.

        Returns:
            Permuted flat tensor ``[B, C, seq_len]``.

        forward method for ImageTopologyLinearizer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C = x.shape[:2]
        x_flat = x.reshape(B, C, -1)
        return x_flat[:, :, self.forward_idx]

    def reverse(self, sequence: torch.Tensor) -> torch.Tensor:
        """Restore original spatial topology from 1D sequence.

        Args:
            sequence: Flat tensor ``[B, C, seq_len]``.

        Returns:
            Restored tensor ``[B, C, *shape]``.
        """
        B, C = sequence.shape[:2]
        x_restored = sequence[:, :, self.inverse_idx]
        return x_restored.reshape(B, C, *self.shape)

    def physical_arc_delta(self) -> torch.Tensor:
        """Per-token arc-length step ``Δ_t = ‖H(s_t) − H(s_{t-1})‖_Ω``.

        Used by ``ContinuousSFCMamba`` to scale the per-token timestep
        in the selective SSM update so the discrete scan is a
        consistent quadrature of the continuous-time SSM along the
        space-filling curve. This is the mechanism behind Theorem A
        (resolution invariance) of ``TODO/Mamba_NO.md``.

        Coordinates are normalised to the unit cube ``[0, 1]^d`` so
        the absolute scale is independent of grid resolution. The
        returned tensor has shape ``(seq_len,)`` with ``Δ_0 = 0`` (no
        previous token).

        Bound (from the Hilbert Hölder property, eq. (1) of
        ``TODO/Mamba_NO.md``)::

            Δ_t ≤ c_d · N^{-1/d}     where c_d = 2·sqrt(d+3)
        """
        # Reconstruct (h, w) or (d, h, w) coordinates from the flat
        # forward_idx, then take the L2 step between consecutive
        # tokens. Normalise by the largest spatial dim so the
        # canonical curve lives in [0, 1]^d.
        coords = np.array(
            np.unravel_index(self.forward_idx.cpu().numpy(), self.shape), dtype=np.float64
        ).T  # (seq_len, d)

        # Normalise to the unit cube — divide each axis by its size.
        scales = np.asarray(self.shape, dtype=np.float64)
        coords_unit = coords / scales[None, :]

        steps = np.linalg.norm(np.diff(coords_unit, axis=0), axis=1)
        delta = np.concatenate([[0.0], steps]).astype(np.float32)
        return torch.from_numpy(delta)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, shape={self.shape}, seq_len={len(self.forward_idx)}"
