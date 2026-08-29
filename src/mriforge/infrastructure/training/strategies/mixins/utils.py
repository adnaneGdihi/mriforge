"""
Utility functions for training strategies and mixins.
"""

import inspect
from typing import Any

# [PERF B1] Module-level cache for signature introspection.
# Keyed on the *underlying function* (not ``id(bound_method)``): bound methods
# are freshly created on every ``getattr(obj, "forward")`` and, once GC'd, their
# ``id()`` can be reused by an unrelated callable — which would make an id-keyed
# cache return a stale accepts/rejects answer (2026-05-31 audit). Keying on the
# function object both de-duplicates across instances of the same class and
# keeps the key alive, so reuse cannot happen.
_CALLABLE_KWARG_CACHE: dict[tuple[Any, str], bool] = {}


def pick_present(*candidates: Any) -> Any:
    """Return the first argument that is not ``None`` (else ``None``).

    SSOT replacement for the ``a or b`` idiom when any operand may be a tensor.
    ``a or b`` evaluates ``bool(a)``, which raises ``RuntimeError: Boolean value
    of Tensor with more than one element is ambiguous`` for any multi-element
    tensor (and silently skips an all-zero / 0-d tensor as "falsy"). Selecting
    purely on ``is None`` avoids both pitfalls.
    """
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _cache_key(callable_obj: Any) -> Any:
    """Stable, GC-safe cache key for a callable (its underlying function)."""
    return getattr(callable_obj, "__func__", callable_obj)


def _callable_accepts_kwarg(callable_obj: Any, argument: str) -> bool:
    """Return True if the callable exposes a keyword argument.

    The helper accounts for callables that accept ``**kwargs`` so we avoid
    brittle signature checks across wrapped modules.

    Results are cached in the module-level ``_CALLABLE_KWARG_CACHE`` dict.
    """
    cache_key = (_cache_key(callable_obj), argument)
    cached = _CALLABLE_KWARG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):  # pragma: no cover - best effort only
        _CALLABLE_KWARG_CACHE[cache_key] = False
        return False

    result = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            result = True
            break
        if parameter.name == argument:
            result = True
            break

    _CALLABLE_KWARG_CACHE[cache_key] = result
    return result


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Helper to safely get config values supporting dot notation and Pydantic models."""
    if config is None:
        return default

    # Handle dot notation for nested keys
    keys = key.split(".")
    current = config

    try:
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
            else:
                current = getattr(current, k, None)

            if current is None:
                return default
        return current
    except (AttributeError, KeyError, TypeError):
        return default


def _scoring_leaf(config: Any, key: str, default: Any = None) -> Any:
    """Read a ``scoring`` leaf, canonical spelling first, flat spelling second.

    ``validation.domain`` and ``validation.output_transform`` moved into a
    ``scoring:`` sub-block. ``RENAMES`` records both moves as **folds**, so a YAML
    declaration in the retired spelling still *loads* -- it just lands under
    ``scoring``. A flat Python read therefore returns ``None`` on every real
    config, and that is how every configured metric transform became inert:
    ``_apply_metric_transforms`` asked for ``output_transform`` at the top level,
    got ``None``, and returned its input unchanged.

    Measured before this helper existed: 145 arms declare ``output_transform``
    (120 of them ``ifft_magnitude``) and not one of them ever had it applied.

    The flat fallback is kept deliberately. It costs nothing on a real config
    (the canonical read answers first) and keeps hand-built stand-ins and dicts
    that spell the leaf flat working -- but it must never be the ONLY read, which
    is the bug this closes.

    Args:
        config: the block that owns the ``scoring`` sub-block (e.g.
            ``config.validation``), a dict, or ``None``.
        key: leaf name inside ``scoring`` -- ``"domain"``, ``"output_transform"``.
        default: returned when neither spelling resolves.
    """
    value = _get_config_value(config, f"scoring.{key}", None)
    if value is None:
        value = _get_config_value(config, key, default)
    return value
