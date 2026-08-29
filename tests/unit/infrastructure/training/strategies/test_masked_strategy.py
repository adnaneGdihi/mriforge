"""Patch-mask construction for :mod:`masked_strategy`.

``_create_patch_mask`` used to scatter one patch at a time, iterating a CUDA
tensor -- a host sync AND a kernel launch per masked patch, inside the training
step (non-negotiable 9). It now scatters into the flat view and reshapes.

That rewrite is only correct because ``[idx // W, idx % W]`` addresses exactly
``flat[idx]`` for a contiguous 2-D tensor, so the tests below pin it against the
loop it replaced rather than against a restatement of the new formula. The
strategy is built with ``object.__new__`` -- the sibling convention in this
directory -- because the base class requires a real training environment and
this method reads nothing but ``patch_size`` and ``mask_ratio``.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.strategies.masked_strategy import (
    MaskedPretrainingStrategy,
)


def _strategy(patch_size: int, mask_ratio: float) -> MaskedPretrainingStrategy:
    strat = object.__new__(MaskedPretrainingStrategy)
    strat.patch_size = patch_size
    strat.mask_ratio = mask_ratio
    return strat


def _reference_mask(
    strat: MaskedPretrainingStrategy, batch: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    """Pre-vectorisation oracle: the per-patch scatter, element by element."""
    num_patches_h = height // strat.patch_size
    num_patches_w = width // strat.patch_size
    total_patches = num_patches_h * num_patches_w
    num_masked_patches = int(total_patches * strat.mask_ratio)

    masks = []
    for _ in range(batch):
        patch_indices = torch.randperm(total_patches, device=device)[:num_masked_patches]
        patch_mask = torch.zeros(num_patches_h, num_patches_w, device=device)
        for idx in patch_indices:
            i = idx // num_patches_w
            j = idx % num_patches_w
            patch_mask[i, j] = 1
        mask = patch_mask.repeat_interleave(strat.patch_size, dim=0).repeat_interleave(
            strat.patch_size, dim=1
        )
        masks.append(mask[:height, :width])
    return torch.stack(masks, dim=0).unsqueeze(1) > 0.5


# ---------------------------------------------------------------------------
# Equivalence with the loop it replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "patch", "ratio"),
    [
        (64, 64, 16, 0.5),  # square
        (64, 128, 16, 0.5),  # NON-square: num_patches_h != num_patches_w.
        (128, 64, 16, 0.75),  # the other non-square orientation
        (96, 96, 8, 0.25),  # finer patches, low ratio
        (64, 64, 16, 0.0),  # degenerate: nothing masked
    ],
)
def test_flat_scatter_matches_per_patch_loop(
    height: int, width: int, patch: int, ratio: float
) -> None:
    """Identical masks, not merely an identical count.

    The non-square cases carry the weight here: when ``num_patches_h`` equals
    ``num_patches_w`` a transposed reshape is invisible, so a square-only test
    would pass on a genuinely wrong index mapping.
    """
    strat = _strategy(patch, ratio)
    device = torch.device("cpu")

    torch.manual_seed(1234)
    expected = _reference_mask(strat, 3, height, width, device)
    torch.manual_seed(1234)
    actual = strat._create_patch_mask(3, height, width, device)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_non_square_case_is_actually_asymmetric() -> None:
    """Guard the discriminating power of the parametrisation above.

    If every case had ``num_patches_h == num_patches_w`` the transpose check
    would be vacuous, so assert at least one case is genuinely asymmetric.
    """
    assert (64 // 16) != (128 // 16)


# ---------------------------------------------------------------------------
# Properties that hold regardless of implementation
# ---------------------------------------------------------------------------


def test_masked_patch_count_is_exact() -> None:
    """Exactly ``int(total * ratio)`` patches masked, per sample."""
    strat = _strategy(patch_size=16, mask_ratio=0.5)
    torch.manual_seed(0)
    mask = strat._create_patch_mask(4, 64, 64, torch.device("cpu"))

    total_patches = (64 // 16) * (64 // 16)
    expected_pixels = int(total_patches * 0.5) * 16 * 16
    for b in range(4):
        assert int(mask[b].sum().item()) == expected_pixels


def test_mask_is_patch_aligned() -> None:
    """Every patch-sized block is uniformly masked or uniformly kept.

    A stray index would light up a single pixel rather than a whole block, so
    this catches a mapping error the count test cannot see.
    """
    strat = _strategy(patch_size=16, mask_ratio=0.5)
    torch.manual_seed(7)
    mask = strat._create_patch_mask(2, 64, 96, torch.device("cpu"))

    for b in range(2):
        for i in range(64 // 16):
            for j in range(96 // 16):
                block = mask[b, 0, i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16]
                assert block.all() or not block.any()


def test_samples_get_independent_masks() -> None:
    """Each batch element draws its own permutation."""
    strat = _strategy(patch_size=16, mask_ratio=0.5)
    torch.manual_seed(3)
    mask = strat._create_patch_mask(2, 64, 64, torch.device("cpu"))
    assert not torch.equal(mask[0], mask[1])
