"""Registry-contract tests: every entry must satisfy the dispatch shape.

Targets ``mriforge.models.registry``. We don't try to instantiate all 400+ models
here (that lives in a future ``@pytest.mark.slow`` exhaustive-instantiation
test, see plan D.3 follow-up). Instead, we verify the structural contract that
``ModelBuilder`` / ``ModelFactory`` / config-audit relies on:

1. The registry is non-empty after auto-discovery has run.
2. Every entry exposes a ``"class"`` key whose value is a Python class.
3. Every entry exposes a ``"mode"`` key (string).
4. ``get_model_class(name)`` returns the registered class for every name.
5. ``get_model_class("definitely-not-a-model")`` raises ``ValueError`` with
   the available-names list (the audit relies on this for misconfigured
   YAMLs — silent fallbacks are forbidden per CLAUDE.md pitfall #9).

Future work (gated ``@pytest.mark.slow``, opt-in nightly): instantiate every
class and run a forward pass. Out of scope here because per-model minimal
config construction is multi-day work.
"""

from __future__ import annotations

import inspect

import pytest

# Trigger auto-discovery. Importing ``init_registry`` only *defines*
# ``populate_model_registry`` — it must be *called* to walk the model tree
# and fill ``MODEL_REGISTRY``. Calling it at module load makes the registry
# populated both for the runtime assertions below and for the collection-time
# ``parametrize(... MODEL_REGISTRY.keys())``. (Previously this relied on some
# earlier test in the session having populated the global registry, so the
# file failed when run in isolation.)
from mriforge.models.init_registry import populate_model_registry
from mriforge.models.registry import MODEL_REGISTRY, get_model_class

populate_model_registry()


def test_registry_is_non_empty_after_discovery() -> None:
    """Auto-discovery should have populated the registry with many models."""
    assert len(MODEL_REGISTRY) >= 50, (
        f"Expected ≥50 registered models post-discovery, got {len(MODEL_REGISTRY)}. "
        "Likely cause: auto-discovery in mriforge.models.init_registry failed silently."
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_entry_exposes_class_field(name: str) -> None:
    """Every registered entry must have a callable class under the 'class' key."""
    entry = MODEL_REGISTRY[name]
    assert "class" in entry, f"Entry '{name}' missing 'class' key: {list(entry.keys())}"
    cls = entry["class"]
    assert inspect.isclass(cls), (
        f"'class' for '{name}' is {type(cls).__name__}, not a class"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_entry_has_mode_string(name: str) -> None:
    """Every entry must declare a training mode (used by strategy dispatch)."""
    entry = MODEL_REGISTRY[name]
    assert "mode" in entry, f"Entry '{name}' missing 'mode' key"
    assert isinstance(entry["mode"], str) and entry["mode"], (
        f"Entry '{name}' has empty/non-string mode: {entry['mode']!r}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_get_model_class_returns_registered_class(name: str) -> None:
    """``get_model_class(name)`` must round-trip the registered class."""
    expected = MODEL_REGISTRY[name]["class"]
    got = get_model_class(name)
    assert got is expected, f"get_model_class({name!r}) returned {got}, expected {expected}"


def test_unknown_model_raises_with_available_names() -> None:
    """Misspelled / dropped model names must fail loud, not silently fallback."""
    with pytest.raises(ValueError, match="not found in registry"):
        get_model_class("definitely-not-a-real-model-name-xyz123")


def test_unknown_model_error_lists_alternatives() -> None:
    """The error message must include the available names so a YAML author can fix the typo."""
    try:
        get_model_class("nonexistent_model_zzz")
    except ValueError as e:
        msg = str(e)
        # At least one real model name should appear in the error.
        sample = next(iter(MODEL_REGISTRY))
        assert sample in msg, (
            f"Expected available-models list in error, got: {msg[:200]}"
        )
    else:
        pytest.fail("Expected ValueError for unknown model name")
