"""Guard the lazy ``sota_registry`` attribute hook in ``mriforge.models``.

``mriforge/models/__init__.py`` used to eagerly ``from . import
sota_registry`` on any ``import mriforge.models`` — pulling 43 model
classes plus timm/torchio/torchmetrics/torchvision/wandb into the CLI
cold-start path. A PEP-562 ``__getattr__`` replaced the eager import.

Since 2026-07-02 ``sota_registry`` is a deprecated no-op shim (all SOTA
registrations are per-class ``@register_model`` decorators fired by the
``populate_model_registry()`` walk), so the lazy hook now guards import
compatibility rather than cold-start cost. These tests assert (1) the
lazy hook still resolves the shim on demand and (2) importing the shim
changes nothing — ``populate_model_registry()`` alone owns the full
``MODEL_REGISTRY`` key set.
"""

from __future__ import annotations

import importlib
import sys


def test_importing_models_is_lean() -> None:
    # The authoritative eager-import check lives in
    # ``tests/unit/cli/test_startup_budget.py`` (clean subprocess). This
    # in-process check just guards the lazy hook contract.
    import mriforge.models as models

    assert hasattr(models, "__getattr__"), "lazy sota_registry hook removed"
    # The attribute must still resolve on demand (import compatibility for
    # legacy callers that reference mriforge.models.sota_registry by name).
    assert models.sota_registry is importlib.import_module(
        "mriforge.models.sota_registry"
    )


def test_populate_registry_key_set_is_stable_across_sota_import() -> None:
    """The retired shim contributes nothing: key set is walk-owned."""
    from mriforge.models.init_registry import populate_model_registry
    from mriforge.models.registry import MODEL_REGISTRY

    populate_model_registry()
    keys_before = set(MODEL_REGISTRY)

    # Importing the shim explicitly (+ calling its no-op entry point +
    # repopulating) must not change the final key set.
    sota = importlib.import_module("mriforge.models.sota_registry")
    sota.register_sota_models()
    populate_model_registry()
    keys_after = set(MODEL_REGISTRY)

    assert keys_after == keys_before
    # Sanity: registration actually happened (catalogue is non-trivial).
    assert len(keys_before) > 10
    assert "mriforge.models.sota_registry" in sys.modules
