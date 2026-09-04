"""Strategy registry contract suite.

One parametric test per key in ``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``.
Contract assertions (all cheap, default-lane):

1. The key resolves to a loadable Python class via ``_load_strategy_class``.
2. The resolved class is a subclass of ``BaseTrainingStrategy``.
3. Every path string is a valid ``module.ClassName`` dotted form.

No instantiation is attempted — building a full ``TrainingEnvironment`` is
a slow/GPU operation.  The subclass check is a pure static import.
"""

from __future__ import annotations

import importlib
import pytest

from tests.utils.registry_iterators import all_strategy_keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_class(class_path: str) -> type:
    """Import and return the class named by *class_path* (dotted form)."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Parametric suite
# ---------------------------------------------------------------------------

@pytest.mark.registry_contract
@pytest.mark.parametrize("key", all_strategy_keys(), ids=lambda k: k)
def test_strategy_key_resolves_to_base_subclass(key: str) -> None:
    """Each STRATEGY_CLASS_PATHS key must load a BaseTrainingStrategy subclass.

    Resolution path mirrors ``TrainingStrategyFactory._load_strategy_class``:
    look up the short key in STRATEGY_CLASS_PATHS, then import the full
    dotted path.
    """
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    full_path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[key]

    # 1. Full path is well-formed (contains a dot)
    assert "." in full_path, (
        f"Strategy '{key}' maps to '{full_path}' which lacks a module separator."
    )

    # 2. The class is importable
    try:
        cls = _resolve_class(full_path)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        pytest.fail(
            f"Strategy '{key}' → '{full_path}' failed to import: {exc}"
        )

    # 3. The class is a BaseTrainingStrategy subclass
    assert issubclass(cls, BaseTrainingStrategy), (
        f"Strategy '{key}' resolves to {cls!r} which is not a subclass of "
        f"BaseTrainingStrategy."
    )
