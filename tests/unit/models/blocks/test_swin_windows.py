"""Windowing primitives: the padding rule, the mask, and the cache (#1345).

The mask's leading dimension is the window count, and ``WindowAttention``
derives ``nW`` from it, so "a mask" is not enough -- it has to be the mask for
the resolution actually being attended over.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.swin_windows import (
    ShiftedWindowMaskCache,
    build_shifted_window_mask,
    padded_resolution,
    window_count,
    window_partition,
    window_reverse,
)


@pytest.mark.parametrize(
    ("height", "width", "window", "expected"),
    [
        (16, 16, 8, (16, 16)),  # already a whole number of windows
        (24, 24, 8, (24, 24)),
        (17, 16, 8, (24, 16)),  # rounds up, one axis at a time
        (7, 7, 7, (7, 7)),
        (8, 8, 7, (14, 14)),
    ],
)
def test_padded_resolution_rounds_up_to_whole_windows(height, width, window, expected):
    """The single spelling of a rule that used to be written twice."""
    assert padded_resolution(height, width, window) == expected


def test_window_count_matches_an_actual_partition():
    """Anti-vacuity: the arithmetic agrees with what ``window_partition`` does."""
    height, width, window = 24, 32, 8
    x = torch.zeros(1, *padded_resolution(height, width, window), 3)
    assert window_partition(x, window).shape[0] == window_count(height, width, window)


def test_partition_round_trips():
    """``window_reverse`` undoes ``window_partition`` exactly."""
    x = torch.randn(2, 16, 24, 5)
    windows = window_partition(x, 8)
    assert torch.equal(window_reverse(windows, 8, 16, 24), x)


@pytest.mark.parametrize(("height", "width"), [(16, 16), (32, 32), (24, 24), (17, 9)])
def test_mask_window_count_follows_the_resolution(height, width):
    """The property #1345 violated."""
    mask = build_shifted_window_mask(height, width, 8, 4)
    assert mask.shape == (window_count(height, width, 8), 64, 64)


def test_mask_is_two_valued_and_lets_a_window_see_itself():
    """0 where attention is allowed, -100 where it is not -- nothing else."""
    mask = build_shifted_window_mask(32, 32, 8, 4)
    assert set(mask.unique().tolist()) <= {0.0, -100.0}
    assert (mask.diagonal(dim1=-2, dim2=-1) == 0.0).all()
    assert (mask == -100.0).any(), "no position is masked, so the mask is inert"


def test_an_unshifted_block_asking_for_a_mask_raises():
    """No silent fallback (non-negotiable 3): shift_size <= 0 is a caller bug."""
    with pytest.raises(ValueError, match="shift_size > 0"):
        build_shifted_window_mask(32, 32, 8, 0)


def test_the_cache_builds_once_per_resolution():
    """Rebuilding every forward would put a Python loop on the warm path."""
    cache = ShiftedWindowMaskCache()
    first = cache.get(32, 32, 8, 4)
    again = cache.get(32, 32, 8, 4)
    assert first is again
    assert len(cache) == 1

    other = cache.get(16, 16, 8, 4)
    assert other is not first
    assert other.shape[0] != first.shape[0]
    assert len(cache) == 2


def test_the_cache_is_bounded():
    """An unbounded dict on a long run is a leak."""
    cache = ShiftedWindowMaskCache(limit=3)
    for size in (8, 16, 24, 32, 40, 48):
        cache.get(size, size, 8, 4)
    assert len(cache) <= 3


def test_the_cache_honours_dtype():
    """A float64 model must not be handed a float32 mask."""
    cache = ShiftedWindowMaskCache()
    assert cache.get(16, 16, 8, 4, dtype=torch.float64).dtype is torch.float64
    assert cache.get(16, 16, 8, 4, dtype=torch.float32).dtype is torch.float32
    assert len(cache) == 2, "dtype is not part of the cache key"
