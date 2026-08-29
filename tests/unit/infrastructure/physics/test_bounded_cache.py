"""Unit tests for :mod:`mriforge.infrastructure.physics.bounded_cache`."""

from __future__ import annotations

import pytest

from mriforge.infrastructure.physics.bounded_cache import (
    DEFAULT_CACHE_CAPACITY,
    BoundedLRUCache,
)


class TestCapacityContract:
    """The cap is the whole point, so it is asserted before anything else."""

    def test_default_capacity_is_positive(self) -> None:
        assert DEFAULT_CACHE_CAPACITY >= 1
        assert BoundedLRUCache().capacity == DEFAULT_CACHE_CAPACITY

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_capacity_raises(self, bad: int) -> None:
        """There is deliberately no spelling for 'unbounded'."""
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            BoundedLRUCache(capacity=bad)

    def test_size_never_exceeds_capacity(self) -> None:
        cache: BoundedLRUCache[int, int] = BoundedLRUCache(capacity=4)
        for i in range(1000):
            cache[i] = i
            assert len(cache) <= 4
        assert len(cache) == 4

    def test_capacity_one_is_legal(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=1)
        cache["a"] = 1
        cache["b"] = 2
        assert len(cache) == 1
        assert "a" not in cache
        assert cache.get("b") == 2


class TestEvictionOrder:
    """Least-recently-USED, not least-recently-inserted."""

    def test_evicts_the_least_recently_inserted_when_never_read(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert "a" not in cache
        assert list(cache) == ["b", "c"]

    def test_read_refreshes_recency(self) -> None:
        """A cascade re-reading one ranking must keep it under single-use traffic."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=2)
        cache["keep"] = 1
        cache["drop"] = 2
        assert cache.get("keep") == 1  # refresh -> 'drop' is now oldest
        cache["new"] = 3
        assert "drop" not in cache
        assert cache.get("keep") == 1

    def test_getitem_also_refreshes_recency(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=2)
        cache["keep"] = 1
        cache["drop"] = 2
        assert cache["keep"] == 1
        cache["new"] = 3
        assert "drop" not in cache

    def test_overwrite_refreshes_without_growing(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=2)
        cache["a"] = 1
        cache["b"] = 2
        cache["a"] = 10
        assert len(cache) == 2
        cache["c"] = 3
        assert "b" not in cache
        assert cache.get("a") == 10

    def test_contains_does_not_refresh_recency(self) -> None:
        """``in`` must be a pure probe, or an eviction test cannot observe anything."""
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=2)
        cache["a"] = 1
        cache["b"] = 2
        assert "a" in cache  # must NOT promote 'a'
        cache["c"] = 3
        assert "a" not in cache


class TestMappingBehaviour:
    def test_get_returns_default_for_missing_key(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        assert cache.get("nope") is None
        assert cache.get("nope", -1) == -1

    def test_getitem_raises_keyerror_for_missing_key(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache()
        with pytest.raises(KeyError):
            _ = cache["nope"]

    def test_clear_empties_the_cache(self) -> None:
        cache: BoundedLRUCache[int, int] = BoundedLRUCache(capacity=4)
        for i in range(4):
            cache[i] = i
        cache.clear()
        assert len(cache) == 0
        assert cache.get(0) is None

    def test_iteration_is_least_to_most_recently_used(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        cache.get("a")
        assert list(cache) == ["b", "c", "a"]

    def test_repr_reports_capacity_and_size(self) -> None:
        cache: BoundedLRUCache[str, int] = BoundedLRUCache(capacity=7)
        cache["a"] = 1
        assert repr(cache) == "BoundedLRUCache(capacity=7, size=1)"
