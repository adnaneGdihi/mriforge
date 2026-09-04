"""Unit tests for _callable_accepts_kwarg caching (Perf B1).

Validates that signature introspection results are cached in the
module-level ``_CALLABLE_KWARG_CACHE`` dict so that subsequent calls
for the same callable + argument bypass ``inspect.signature()``.
"""

from spectramr.infrastructure.training.strategies.base import (
    _CALLABLE_KWARG_CACHE,
    _callable_accepts_kwarg,
)

# ---------------------------------------------------------------------------
# Test fixtures: simple callables with known signatures
# ---------------------------------------------------------------------------


def _fn_with_kwarg(x: int, timesteps: int = 0) -> int:
    return x + timesteps


def _fn_with_var_kwargs(x: int, **kwargs) -> int:
    return x


def _fn_without_kwarg(x: int, y: int = 0) -> int:
    return x + y


class _ModelLike:
    """Mimics a nn.Module's forward signature."""

    def forward(self, x, timesteps=None, mask=None):
        return x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCallableAcceptsKwargCaching:
    """Verify caching behaviour of _callable_accepts_kwarg."""

    def setup_method(self):
        """Clear the cache before each test to avoid cross-contamination."""
        _CALLABLE_KWARG_CACHE.clear()

    # ---- Correctness ---------------------------------------------------

    def test_detects_explicit_kwarg(self):
        assert _callable_accepts_kwarg(_fn_with_kwarg, "timesteps") is True

    def test_detects_var_kwargs(self):
        assert _callable_accepts_kwarg(_fn_with_var_kwargs, "anything") is True

    def test_rejects_missing_kwarg(self):
        assert _callable_accepts_kwarg(_fn_without_kwarg, "timesteps") is False

    def test_detects_method_kwarg(self):
        model = _ModelLike()
        assert _callable_accepts_kwarg(model.forward, "timesteps") is True
        assert _callable_accepts_kwarg(model.forward, "mask") is True
        assert _callable_accepts_kwarg(model.forward, "nonexistent") is False

    # ---- Caching -------------------------------------------------------

    def test_cache_populated_after_first_call(self):
        assert len(_CALLABLE_KWARG_CACHE) == 0
        _callable_accepts_kwarg(_fn_with_kwarg, "timesteps")
        assert len(_CALLABLE_KWARG_CACHE) == 1

    def test_cache_returns_same_result(self):
        first = _callable_accepts_kwarg(_fn_with_kwarg, "timesteps")
        second = _callable_accepts_kwarg(_fn_with_kwarg, "timesteps")
        assert first == second is True
        # Should still be a single cache entry
        assert len(_CALLABLE_KWARG_CACHE) == 1

    def test_cache_keyed_by_id_and_argument(self):
        _callable_accepts_kwarg(_fn_with_kwarg, "timesteps")
        _callable_accepts_kwarg(_fn_with_kwarg, "x")
        # Two different arguments → two cache entries
        assert len(_CALLABLE_KWARG_CACHE) == 2

    def test_cache_handles_different_callables(self):
        _callable_accepts_kwarg(_fn_with_kwarg, "timesteps")
        _callable_accepts_kwarg(_fn_without_kwarg, "timesteps")
        assert len(_CALLABLE_KWARG_CACHE) == 2

    def test_false_result_is_also_cached(self):
        result = _callable_accepts_kwarg(_fn_without_kwarg, "timesteps")
        assert result is False
        # Keyed on the underlying function object (NOT id(), which could be
        # reused by an unrelated callable after GC — 2026-05-31 fix), plus
        # the argument name. A plain function has no __func__, so the key is
        # the function itself.
        key = (_fn_without_kwarg, "timesteps")
        assert key in _CALLABLE_KWARG_CACHE
        assert _CALLABLE_KWARG_CACHE[key] is False
