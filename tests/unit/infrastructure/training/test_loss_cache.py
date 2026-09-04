"""Unit tests for the loss-module cache singleton.

Targets ``spectramr.infrastructure.training.loss_cache.LossModuleCache``.

Properties verified:
- Singleton guarantee: multiple ``LossModuleCache()`` calls return the same
  object.
- ``put`` / ``get`` / ``contains`` / ``size`` are consistent.
- ``clear`` resets the cache to empty.
- ``keys``, ``values``, ``items`` mirror the dict interface.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_cache():
    """Return the singleton but clear it first so tests are isolated."""
    from spectramr.infrastructure.training.loss_cache import LossModuleCache

    cache = LossModuleCache()
    cache.clear()
    return cache


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


def test_canary_singleton() -> None:
    from spectramr.infrastructure.training.loss_cache import LossModuleCache

    a = LossModuleCache()
    b = LossModuleCache()
    assert a is b, "LossModuleCache must be a singleton"


# ---------------------------------------------------------------------------
# Parametrized: put/get round-trip for different value types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("str_val", "hello"),
        ("int_val", 42),
        ("dict_val", {"a": 1}),
        ("none_val", None),
    ],
)
def test_put_get_roundtrip(key: str, value) -> None:
    cache = _fresh_cache()
    cache.put(key, value)
    assert cache.get(key) == value


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def test_get_missing_returns_none() -> None:
    cache = _fresh_cache()
    assert cache.get("__nonexistent__") is None


def test_contains_true_after_put() -> None:
    cache = _fresh_cache()
    cache.put("k", "v")
    assert cache.contains("k")


def test_contains_false_before_put() -> None:
    cache = _fresh_cache()
    assert not cache.contains("__never_put__")


def test_size_increments() -> None:
    cache = _fresh_cache()
    assert cache.size() == 0
    cache.put("a", 1)
    assert cache.size() == 1
    cache.put("b", 2)
    assert cache.size() == 2


def test_clear_resets_to_zero() -> None:
    cache = _fresh_cache()
    cache.put("a", 1)
    cache.put("b", 2)
    cache.clear()
    assert cache.size() == 0
    assert cache.get("a") is None


def test_keys_values_items_consistency() -> None:
    cache = _fresh_cache()
    cache.put("x", 10)
    cache.put("y", 20)
    keys = cache.keys()
    values = cache.values()
    items = cache.items()
    assert set(keys) == {"x", "y"}
    assert set(values) == {10, 20}
    assert set(items) == {("x", 10), ("y", 20)}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_edge_overwrite_same_key() -> None:
    cache = _fresh_cache()
    cache.put("k", "first")
    cache.put("k", "second")
    assert cache.get("k") == "second"
    assert cache.size() == 1


def test_edge_empty_string_key() -> None:
    cache = _fresh_cache()
    cache.put("", "empty_key")
    assert cache.contains("")
    assert cache.get("") == "empty_key"


# ---------------------------------------------------------------------------
# get_instance class-method alias
# ---------------------------------------------------------------------------


def test_get_instance_returns_singleton() -> None:
    from spectramr.infrastructure.training.loss_cache import LossModuleCache

    inst = LossModuleCache.get_instance()
    assert inst is LossModuleCache()
