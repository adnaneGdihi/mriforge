"""A capacity-bounded LRU memo for seed-keyed accelerator caches.

Why this exists rather than a plain ``dict``
--------------------------------------------
A mask accelerator memoizes its expensive draw on ``(shape, device, seed)``.
The seed **must** be in that key: ``_generate_batch_masks_dynamic``
(``models/diffusion/kspace_process.py:1172``) reuses ONE accelerator instance
and mutates ``.seed`` per sample, so a shape-only key hands every sample the
first draw and silently makes ``enable_dynamic_mask`` a no-op.

Putting the seed in the key fixes that and creates the opposite failure: the
key is then unbounded, one permanent entry accrues per sample per iteration,
and nothing ever evicts. Because the dynamic path never asks for the same seed
twice, those entries are **pure garbage** -- the hit rate on that path is zero
while the growth is linear in iterations.

So the two properties are a pair, and this class is the second half:

* the seed stays in the key, so the fixed-seed cascade (reverse trajectory,
  validation) still gets its exact-reuse hit, which is the only place the
  memo was ever load-bearing;
* capacity is bounded, so the dynamic path costs O(capacity) instead of
  O(iterations x batch).

``lru_cache`` does not fit: these caches hang off a mutable instance whose
``.seed`` changes underneath them, the values are tensors that must be dropped
when the instance is, and callers need ``len()`` to assert the bound in tests.

Not thread-safe, matching the ``dict`` it replaces. Accelerators are used from
the dataloader worker or the training thread that owns them, never shared.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterator

__all__ = ["DEFAULT_CACHE_CAPACITY", "BoundedLRUCache"]

#: Entries retained by default.
#:
#: Sized for the *fixed-seed* workload, which is the only one that reads back
#: what it wrote: one entry per ``(shape, device)`` the run touches, and a run
#: touches a handful. The dynamic-mask workload never hits, so for it any cap
#: is equally correct and a small one is strictly cheaper -- a
#: ``DensityNestedKSpaceAccelerator`` 2D ranking at 256x256 is one int64 index
#: per bin (~512 KiB), so 32 entries bound the worst case near 16 MiB where the
#: unbounded dict grew without limit for the life of the run.
DEFAULT_CACHE_CAPACITY = 32


class BoundedLRUCache[K: Hashable, V]:
    """Least-recently-used mapping with a hard entry cap.

    Reads and writes both mark an entry as recently used, so a cascade that
    re-reads one ranking every timestep keeps it regardless of how much
    single-use traffic flows past it.

    Args:
        capacity: Maximum entries retained. Must be >= 1; there is no
            "unbounded" spelling on purpose, since that is the defect this
            class exists to make unreachable.

    Raises:
        ValueError: If ``capacity`` is not a positive integer.
    """

    __slots__ = ("_capacity", "_data")

    def __init__(self, capacity: int = DEFAULT_CACHE_CAPACITY) -> None:
        capacity = int(capacity)
        if capacity < 1:
            msg = f"capacity must be >= 1 (got {capacity}); an unbounded cache is the defect this class replaces."
            raise ValueError(msg)
        self._capacity = capacity
        self._data: OrderedDict[K, V] = OrderedDict()

    @property
    def capacity(self) -> int:
        """Maximum number of retained entries."""
        return self._capacity

    def get(self, key: K, default: V | None = None) -> V | None:
        """Return the value for ``key``, refreshing its recency, else ``default``."""
        try:
            value = self._data[key]
        except KeyError:
            return default
        self._data.move_to_end(key)
        return value

    def __getitem__(self, key: K) -> V:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        """Membership WITHOUT refreshing recency, so a test can probe eviction."""
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[K]:
        """Iterate keys least- to most-recently used."""
        return iter(self._data)

    def clear(self) -> None:
        """Drop every entry."""
        self._data.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(capacity={self._capacity}, size={len(self._data)})"
