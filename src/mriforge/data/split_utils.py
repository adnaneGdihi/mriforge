"""Deterministic file-level train/val index splitting (data SSOT).

One honest split helper shared by every dataset instantiator and manifest
loader so the single-file edge case is handled identically everywhere instead
of drifting into three different wrong behaviours (silent leak, empty train,
warn-and-leak). File-level (never slice-level) splitting avoids the subtler
leak where slices of the same volume land in both train and val.

Policy:

- ``validation_split <= 0`` -> train-only: every item is training, validation
  is empty. This is the explicit escape hatch for single-file smoke / overfit
  sanity runs; set ``validation_split: 0`` in the config to opt in.
- exactly one item with ``validation_split > 0`` -> **raise**. A one-file
  corpus cannot form a train/val split that does not overlap: reusing the file
  for both leaks validation into training, and holding it out empties the
  training set. The caller must supply >= 2 files or choose the train-only
  escape hatch above.
- otherwise -> deterministic non-overlapping split with the last ``n_val``
  items as validation, clamped so ``1 <= n_val <= n - 1`` (both splits
  non-empty).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["split_index"]


def split_index[T](items: Sequence[T], validation_split: float) -> tuple[list[T], list[T]]:
    """Split an ordered index into non-overlapping ``(train, val)`` lists.

    Args:
        items: the ordered index (file records, paths, ...). Copied, never
            mutated.
        validation_split: fraction of ``items`` to hold out for validation.
            ``<= 0`` selects the train-only escape hatch.

    Returns:
        ``(train_items, val_items)`` — disjoint lists whose union is ``items``.

    Raises:
        ValueError: exactly one item with ``validation_split > 0`` (no
            non-overlapping split exists; see module docstring).
    """
    seq = list(items)
    n = len(seq)
    if n == 0:
        return [], []
    if validation_split <= 0.0:
        return seq, []
    if n == 1:
        raise ValueError(
            "Cannot form a non-overlapping train/val split from a single file "
            f"with validation_split={validation_split!r}: reusing it for both "
            "leaks validation into training, and holding it out empties the "
            "training set. Provide >= 2 files, or set validation_split: 0 for "
            "an explicit train-only run."
        )
    n_val = min(max(1, round(n * validation_split)), n - 1)
    return seq[:-n_val], seq[-n_val:]
