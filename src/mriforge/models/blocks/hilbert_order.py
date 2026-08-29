"""HilbertOrder — canonical SFC API for the traversal-based program.

Implements the API contract from the *Traversal-Based Computational MRI*
implementation specification (§6.3, §14.2): a precomputed Hilbert
permutation registered as an ``nn.Buffer`` with O(1) GPU lookup, plus
``linearize()`` and ``unflatten()`` helpers that every downstream block
(HSSC, TALL, DTN2S, XDTA) depends on.

This is a thin, stable wrapper around the lower-level functions in
:mod:`mriforge.models.blocks.topology_linearizer` (Hilbert/Morton/snake/
zigzag/radial 2-D + 3-D), exposing them in the canonical
``permutation`` / ``inverse`` form the spec mandates.

Usage::

    h = HilbertOrder(order=9, n_dims=2, device="cuda")  # 512x512
    seq = h.linearize(image)                            # (B, C, N)
    img = h.unflatten(seq, (512, 512))                  # (B, C, 512, 512)

References
----------
* J. Skilling, "Programming the Hilbert curve," AIP Conf. Proc. 707,
  pp. 381–387, 2004. (canonical generator; §39 in IMPLEMENTATION_SPEC.md)
* B. Moon et al., "Analysis of the clustering properties of the Hilbert
  space-filling curve," IEEE TKDE 13(1), 2001. (locality bound used by
  Phase 1 validation; §4 in IMPLEMENTATION_SPEC.md)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn as nn

from mriforge.models.blocks.topology_linearizer import (
    _morton_2d_indices,
    _morton_3d_indices,
    _raster_indices,
    _rect_hilbert_2d_indices,
    _rect_hilbert_3d_indices,
    _snake_2d_indices,
    _zigzag_2d_indices,
)

logger = logging.getLogger(__name__)


# Hilbert entries use the resolution-agnostic ``*_rect`` builders: they reduce
# byte-identically to the strict square/cubic power-of-2 builders but do not
# assert on non-power-of-2 / non-square extents (which crashed HilbertOrder at
# construction — e.g. xdta's default image_size=320). Morton / snake / zigzag /
# raster carry no power-of-2 constraint.
_BUILDERS = {
    ("hilbert", 2): _rect_hilbert_2d_indices,
    ("hilbert", 3): _rect_hilbert_3d_indices,
    ("morton", 2): _morton_2d_indices,
    ("morton", 3): _morton_3d_indices,
    ("snake", 2): _snake_2d_indices,
    ("zigzag", 2): _zigzag_2d_indices,
    ("raster", 2): lambda h, w: _raster_indices(h, w),
    ("raster", 3): lambda d, h, w: _raster_indices(d, h, w),
}


class HilbertOrder(nn.Module):
    """Precomputed space-filling-curve traversal for a fixed cubic grid.

    The default mode is Hilbert (per the spec); other curves are
    available via ``mode`` for ablation studies (Phase 1 §7.1 S1.1
    cluster-count comparison).

    Args:
        order:  Recursion depth ``p`` such that ``side = 2 ** p``.
                Used only when ``shape`` is None. Must be ≥ 1.
        n_dims: 2 (image) or 3 (volume). Default 2.
        device: Target device for the cached permutation buffer.
        mode:   One of ``"hilbert"``, ``"morton"``, ``"snake"``,
                ``"zigzag"``, ``"raster"``. Default ``"hilbert"``.
        shape:  Optional explicit shape ``(H, W)`` or ``(D, H, W)``.
                When provided, ``order`` is ignored and the curve is
                built for the (possibly non-power-of-2) shape.

    Buffers:
        permutation: ``(N,)`` int64 tensor — ``permutation[i]`` is the
                     row-major linear index of the ``i``-th voxel along
                     the curve.
        inverse:     ``(N,)`` int64 tensor — inverse permutation.
    """

    def __init__(
        self,
        order: int | None = None,
        n_dims: int = 2,
        device: str | torch.device = "cpu",
        mode: str = "hilbert",
        shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if shape is None and order is None:
            raise ValueError("HilbertOrder: either `order` or `shape` must be provided")
        if shape is None:
            side = 2 ** int(order)
            shape = (side,) * n_dims
        else:
            shape = tuple(int(s) for s in shape)
            n_dims = len(shape)
        if n_dims not in (2, 3):
            raise ValueError(f"HilbertOrder supports n_dims in (2, 3); got {n_dims}")
        if (mode, n_dims) not in _BUILDERS:
            raise ValueError(
                f"HilbertOrder: unknown mode/n_dims combo "
                f"({mode!r}, {n_dims}). Valid: {list(_BUILDERS.keys())}"
            )

        self.order = order
        self.n_dims = n_dims
        self.shape = shape
        self.mode = mode
        self.n_points = int(torch.prod(torch.tensor(shape)).item())
        self.side = shape[-1]  # canonical side for cubic grids

        # Build permutation on CPU then move. All builders here are
        # resolution-agnostic (hilbert -> *_rect; morton/snake/zigzag/raster
        # carry no power-of-2 constraint), so any (H, W) / (D, H, W) is accepted.
        builder = _BUILDERS[(mode, n_dims)]
        perm_np = builder(*shape)

        perm = torch.as_tensor(perm_np, dtype=torch.long)
        if perm.numel() != self.n_points:
            raise RuntimeError(
                f"HilbertOrder: builder returned {perm.numel()} elements, "
                f"expected {self.n_points} for shape {shape}"
            )
        # Count guard only. A true bijection guard is impossible here: every
        # builder ends in ``np.argsort(...)``, whose output is *always* a
        # permutation of ``range(N)`` — a sort-based check can never fail. The
        # numel guard above is the real invariant; locality is pinned by the
        # dedicated topology_linearizer tests.

        inv = torch.argsort(perm)
        self.register_buffer("permutation", perm.to(device))
        self.register_buffer("inverse", inv.to(device))

    # ──────────────────────────────────────────────────────────────────
    # Core operations
    # ──────────────────────────────────────────────────────────────────

    def linearize(self, x: torch.Tensor) -> torch.Tensor:
        """Permute the trailing ``n_dims`` axes of ``x`` into curve order.

        Args:
            x: Tensor of shape ``(B, C, *spatial)`` where ``spatial``
               equals ``self.shape``. Real or complex.

        Returns:
            Tensor of shape ``(B, C, N)`` along the curve.
        """
        if x.shape[-self.n_dims :] != tuple(self.shape):
            raise ValueError(
                f"HilbertOrder.linearize: expected trailing shape "
                f"{tuple(self.shape)}, got {tuple(x.shape[-self.n_dims :])}"
            )
        flat = x.flatten(start_dim=-self.n_dims)
        return flat.index_select(dim=-1, index=self.permutation)

    def unflatten(
        self,
        seq: torch.Tensor,
        spatial_shape: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Inverse of :meth:`linearize` — restore trailing spatial dims."""
        if spatial_shape is None:
            spatial_shape = tuple(self.shape)
        spatial_shape = tuple(int(s) for s in spatial_shape)
        flat = seq.index_select(dim=-1, index=self.inverse)
        return flat.unflatten(-1, spatial_shape)

    def __repr__(self) -> str:
        return (
            f"HilbertOrder(mode={self.mode!r}, shape={tuple(self.shape)}, "
            f"n_points={self.n_points}, device={self.permutation.device})"
        )


__all__ = ["HilbertOrder"]
