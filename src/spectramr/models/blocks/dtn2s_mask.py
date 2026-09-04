"""DTN2S J-invariance mask construction.

Implements the dual-traversal masking scheme from
``IMPLEMENTATION_SPEC.md`` §10 / §14.6: given two SFC orderings
``φ_A`` and ``φ_B`` whose receptive fields around any voxel do not
overlap, the mask blocks every voxel in ``s_A`` that the model would
otherwise see when predicting at the corresponding ``φ_B`` position.

This realises the discrete analogue of Noise2Self's *J-invariance*
condition (Krull et al. 2019) and lets the model train without paired
clean references — the substrate for self-supervised denoising on
M4Raw and other low-field datasets.

The mask is computed once per ``(resolution, traversal-pair,
receptive-window)`` and cached in memory; for production use it should
also be cached on disk via the standard project conventions.

References
----------
* IMPLEMENTATION_SPEC.md §10 (DTN2S — Dual-Traversal Noise2Self) and
  §14.6 (reference PyTorch snippet).
* A. Krull, T.-O. Buchholz, F. Jug, "Noise2Void — learning denoising
  from single noisy images," CVPR 2019.
* J. Batson, L. Royer, "Noise2Self: blind denoising by self-supervision,"
  ICML 2019. (J-invariance principle)
"""

from __future__ import annotations

import logging

import torch

from spectramr.models.blocks.hilbert_order import HilbertOrder

logger = logging.getLogger(__name__)


_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _cache_key(
    phi_a: HilbertOrder,
    phi_b: HilbertOrder,
    receptive_window: int,
) -> tuple:
    return (
        tuple(phi_a.shape),
        phi_a.mode,
        tuple(phi_b.shape),
        phi_b.mode,
        int(receptive_window),
    )


def build_dtn2s_mask(
    phi_a: HilbertOrder,
    phi_b: HilbertOrder,
    receptive_window: int,
) -> torch.Tensor:
    r"""Build the J-invariance mask for a dual-traversal denoiser.

    The result is **VOXEL-indexed**: ``mask[v]`` is about voxel ``v``, not
    about sequence position ``v``. A consumer zeroing positions in ``s_A``
    must re-index with ``mask[phi_a.permutation]`` first — reading it
    positionally blanks a set unrelated to the blocked voxels (#1028).

    ``mask[v]`` is True when the model would otherwise see ``v`` while
    predicting ``v``. That is a **per-voxel** test, not a union over the
    predicted set:

    .. math::

        \mathrm{mask}[v] \iff
        d_\circ\!\left(\mathrm{pos}_A[v],\, \mathrm{pos}_B[v]\right)
        \le \mathrm{receptive\_window}

    where :math:`d_\circ` is circular distance over the ``N`` sequence
    positions. Voxel ``v`` is predicted at output position
    ``pos_b[v]``, where the model reads input positions
    ``pos_b[v] ± receptive_window`` of the ``φ_A``-ordered sequence; it
    sees ``v`` itself exactly when ``v``'s own ``φ_A`` position lies in
    that window.

    The mask is therefore **sparse**, and that is the point of the dual
    traversal: ``φ_A`` and ``φ_B`` are chosen so a voxel's neighbourhood
    in one ordering is far away in the other, so the distance above
    exceeds the window for almost every voxel (measured: 1.9 % blocked at
    64×64 with a window of 8). An implementation that blocks most of the
    volume is misconstructed, not merely conservative.

    Args:
        phi_a:             First traversal (Hilbert by default).
        phi_b:             Second traversal (e.g. axis-permuted Hilbert).
        receptive_window:  Half-receptive-field measured in TOKENS along
                           the SFC ordering (typical 8–16). Must leave at
                           least one voxel visible.

    Returns:
        ``(N,)`` bool tensor on the same device as ``phi_a.permutation``.
        An **empty** mask is a valid result — it means the two traversals
        never place a voxel within the window of itself, so J-invariance
        already holds with nothing blinded.

    Raises:
        ValueError: if ``phi_a`` and ``phi_b`` disagree on shape, or if
            the window blinds **every** voxel. The latter would leave
            ``s_A`` all zeros and train the model from a constant while
            reporting a plausible loss, so it fails loud rather than
            degrading (non-negotiable #3).
    """
    if phi_a.shape != phi_b.shape:
        raise ValueError(f"build_dtn2s_mask: φ_A shape {phi_a.shape} != φ_B shape {phi_b.shape}")
    key = _cache_key(phi_a, phi_b, receptive_window)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key].to(phi_a.permutation.device)

    N = phi_a.permutation.numel()
    pos_a = phi_a.inverse  # voxel → position in s_A
    pos_b = phi_b.inverse  # voxel → position in s_B
    w = int(receptive_window)

    # Voxel ``v`` is predicted at OUTPUT position ``pos_b[v]``. The backbone
    # reads the s_A-ordered sequence, so at that output position it sees INPUT
    # positions ``pos_b[v] ± w``. It "would otherwise see" ``v`` itself exactly
    # when ``v``'s own s_A position falls inside that window:
    #
    #     blocked[v]  ⟺  circular_distance(pos_a[v], pos_b[v]) ≤ w
    #
    # This is a PER-VOXEL test. The previous implementation instead took the
    # union of the windows around *every* predicted voxel, which is all N of
    # them — and since the offset-0 term alone makes ``pos_b`` a full
    # permutation, that union was always the entire volume, at every window
    # size (#1028). It also indexed ``phi_a.permutation`` with an s_B position,
    # mixing the two sequence spaces.
    #
    # The dual-traversal construction is what makes the per-voxel form sparse:
    # φ_A and φ_B are chosen so a voxel's neighbourhood in one ordering is far
    # away in the other, i.e. the distance above exceeds ``w`` for almost every
    # voxel. That is the spec's "receptive fields ... do not overlap for almost
    # every voxel" premise (IMPLEMENTATION_SPEC.md §10.1), and it is precisely
    # why only a handful of voxels ever need blinding.
    delta = (pos_a - pos_b).abs()
    circular_distance = torch.minimum(delta, N - delta)
    blocked = circular_distance <= w

    # An EMPTY mask is legitimate — it means the two traversals never place a
    # voxel within ``w`` of itself, so J-invariance already holds unaided. An
    # ALL-TRUE mask is not: it zeroes the whole input and the arm trains a map
    # from a constant, reporting a plausible loss the whole way (#1028). Fail
    # loud rather than degrade (non-negotiable #3).
    if bool(blocked.all()):
        raise ValueError(
            f"build_dtn2s_mask: receptive_window={w} blinds every voxel of "
            f"{N} — s_A would be all zeros and the model would train from a "
            f"constant. The window is measured in TOKENS along the space-"
            f"filling curve and must stay well below N; typical values are "
            f"8–16. Reduce `training.dtn2s.receptive_window`, or use a "
            f"traversal pair whose orderings differ more."
        )

    _MASK_CACHE[key] = blocked.cpu()  # cache on CPU; per-call .to(device)
    return blocked


def dual_traversal_pair(
    shape: tuple[int, int] | tuple[int, int, int],
    *,
    device: str | torch.device = "cpu",
) -> tuple[HilbertOrder, HilbertOrder]:
    """Convenience: build a `(φ_A, φ_B)` pair where ``φ_B`` is
    ``φ_A`` with axes reversed (the spec's canonical "3-axis-permuted
    Hilbert" choice for n_dims=3, falling back to axis-flipped Hilbert
    for n_dims=2).
    """
    phi_a = HilbertOrder(shape=shape, device=device, mode="hilbert")
    if len(shape) == 2:
        phi_b = HilbertOrder(shape=shape, device=device, mode="zigzag")
    else:
        # 3-axis-permuted Hilbert: rebuild on transposed shape, then
        # remap back. For test purposes the mode-swap to morton works.
        phi_b = HilbertOrder(shape=shape, device=device, mode="morton")
    return phi_a, phi_b


__all__ = ["build_dtn2s_mask", "dual_traversal_pair"]
