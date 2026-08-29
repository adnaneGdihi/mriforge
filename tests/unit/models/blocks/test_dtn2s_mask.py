"""Tests for ``build_dtn2s_mask`` and ``dual_traversal_pair``.

Targets ``mriforge.models.blocks.dtn2s_mask`` (DTN2S J-invariance mask).

Categories:

- Shape mismatch between φ_A and φ_B raises
- ``build_dtn2s_mask`` returns an ``[N]`` bool tensor
- ``receptive_window=0`` blocks exactly the φ_B-corresponding voxels
- Larger ``receptive_window`` blocks at least as many voxels as smaller one
- Mask is deterministic (same key → identical result; cache works)
- ``dual_traversal_pair`` returns two distinct ``HilbertOrder`` instances
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.blocks.dtn2s_mask import build_dtn2s_mask, dual_traversal_pair
from mriforge.models.blocks.hilbert_order import HilbertOrder


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_shape_mismatch_rejected() -> None:
    """Mismatched φ_A / φ_B shapes raise."""
    phi_a = HilbertOrder(shape=(4, 4))
    phi_b = HilbertOrder(shape=(8, 8))
    with pytest.raises(ValueError, match="shape"):
        build_dtn2s_mask(phi_a, phi_b, receptive_window=2)


# ---------------------------------------------------------------------------
# Mask shape & dtype
# ---------------------------------------------------------------------------


def test_mask_shape_and_dtype() -> None:
    """Mask has shape ``[N]`` and dtype bool, where N is the voxel count."""
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    mask = build_dtn2s_mask(phi_a, phi_b, receptive_window=2)
    assert mask.shape == (16,)
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# Receptive-window monotonicity
# ---------------------------------------------------------------------------


def test_larger_window_blocks_at_least_as_many() -> None:
    """Increasing ``receptive_window`` cannot un-block voxels."""
    phi_a, phi_b = dual_traversal_pair(shape=(8, 8))
    small = build_dtn2s_mask(phi_a, phi_b, receptive_window=1)
    big = build_dtn2s_mask(phi_a, phi_b, receptive_window=4)
    # Every voxel blocked by the small window is blocked by the big one.
    assert ((small & ~big).sum().item()) == 0


def test_zero_window_blocks_only_coincident_voxels() -> None:
    """``receptive_window=0`` → blocks exactly the traversals' fixed points.

    Replaces ``test_zero_window_blocks_only_corresponding_voxels`` (#1028),
    which asserted ``mask.all()``. Its NAME was the intended contract; its body
    reasoned its way to all-True and accepted it, so the suite pinned the defect
    as expected behaviour rather than catching it.

    At ``recv=0`` the model's window is the single output position, so voxel
    ``v`` is visible when predicting itself iff its two sequence positions
    coincide: ``pos_a[v] == pos_b[v]``. That set is usually small and may be
    EMPTY -- empty is a legitimate mask (nothing needs blinding), all-True is not.
    """
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    mask = build_dtn2s_mask(phi_a, phi_b, receptive_window=0)
    expected = phi_a.inverse == phi_b.inverse
    assert torch.equal(mask, expected)
    assert not mask.all().item(), "all-True is the degenerate case #1028 fixed"


def test_mask_is_sparse_not_degenerate() -> None:
    """The whole point: some context must survive for the model to read.

    A mask that blocks everything makes ``s_a_masked`` all zeros, so the arm
    trains a map from a constant. This is the assertion the old implementation
    could not pass at any window size.
    """
    for shape in [(4, 4), (8, 8), (16, 16)]:
        for recv in (0, 1, 2, 4):
            phi_a, phi_b = dual_traversal_pair(shape=shape)
            frac = build_dtn2s_mask(phi_a, phi_b, recv).float().mean().item()
            assert frac < 1.0, (
                f"shape={shape} recv={recv}: {frac:.1%} of voxels blocked -- "
                "no visible context remains"
            )


def test_receptive_window_actually_changes_the_mask() -> None:
    """``receptive_window`` must be a live knob, not accepted and ignored.

    Under the old construction every value produced an identical all-True mask,
    so the knob was inert -- pitfall #15 hiding inside a correctness bug.
    """
    phi_a, phi_b = dual_traversal_pair(shape=(16, 16))
    fracs = [
        build_dtn2s_mask(phi_a, phi_b, r).float().mean().item() for r in (0, 2, 8)
    ]
    assert fracs == sorted(fracs), f"not monotone in receptive_window: {fracs}"
    assert len(set(fracs)) > 1, f"window had no effect at all: {fracs}"


def test_degenerate_mask_raises_rather_than_training_on_zeros() -> None:
    """A window wide enough to blind everything must fail loud (#1028, #3).

    With ``recv >= N`` every position falls inside every window, so the mask
    would be all-True. Silently returning it is what produced an arm training
    from constant zeros; raising converts that into a visible config error.
    """
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    n = phi_a.permutation.numel()
    with pytest.raises(ValueError, match="every voxel"):
        build_dtn2s_mask(phi_a, phi_b, receptive_window=n)


# ---------------------------------------------------------------------------
# Cache determinism
# ---------------------------------------------------------------------------


def test_mask_deterministic() -> None:
    """Same inputs → same output (cached or not)."""
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    m1 = build_dtn2s_mask(phi_a, phi_b, receptive_window=2)
    m2 = build_dtn2s_mask(phi_a, phi_b, receptive_window=2)
    assert torch.equal(m1, m2)


# ---------------------------------------------------------------------------
# dual_traversal_pair convenience
# ---------------------------------------------------------------------------


def test_dual_traversal_pair_returns_distinct_orderings_2d() -> None:
    """For 2-D shape the two traversals are different."""
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    # Permutations should differ at *some* position.
    assert not torch.equal(phi_a.permutation, phi_b.permutation)


def test_dual_traversal_pair_returns_hilbert_order_instances() -> None:
    """Both elements of the returned tuple are ``HilbertOrder``."""
    phi_a, phi_b = dual_traversal_pair(shape=(4, 4))
    assert isinstance(phi_a, HilbertOrder)
    assert isinstance(phi_b, HilbertOrder)


def test_dual_traversal_pair_3d_supported() -> None:
    """3-D shape is also supported (morton fallback)."""
    phi_a, phi_b = dual_traversal_pair(shape=(2, 2, 2))
    assert phi_a.shape == (2, 2, 2)
    assert phi_b.shape == (2, 2, 2)
