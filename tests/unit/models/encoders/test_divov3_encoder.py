"""Guard that ``divov3_encoder`` imports.

It did not, for an unknown length of time. Line 20 read
``from ...utils.metaclass import ModuleABCMeta`` -- a relative import resolving to
``mriforge.utils.metaclass``, a package that has not existed since the 2026-05
``src`` -> ``mriforge`` refactor (the metaclass lives at
``mriforge.shared.utils.metaclass``, which is what every other caller imports).

Nothing noticed because the model-discovery walk swallowed the ImportError at DEBUG
(``init_registry``), so the module simply vanished from the catalog on every run. The
walk now WARNs and records the failure, which is how this surfaced. The cheapest test
that can fail if the import regresses is the import itself.
"""

from __future__ import annotations

import importlib


def test_the_module_imports() -> None:
    """A module that cannot be imported registers nothing and is dead weight."""
    mod = importlib.import_module("mriforge.models.encoders.divov3_encoder")
    assert mod is not None


def test_the_discovery_walk_reports_no_import_failures() -> None:
    """The model catalog must be complete: no module may drop out silently.

    A non-empty failure map means some ``@register_model`` never fired, so a valid
    ``model_type`` can be rejected as unknown -- which is exactly how the CI yaml-audit
    came to reject ``neural_complex_sum``.

    ``force=True`` so the walk actually re-runs: without it the idempotency flag makes
    this a no-op and the assertion would grade whatever ``_IMPORT_FAILURES`` happened to
    hold from an earlier test, not a real discovery pass.
    """
    from mriforge.models.init_registry import (
        get_registry_import_failures,
        populate_model_registry,
    )

    populate_model_registry(force=True)
    failures = get_registry_import_failures()
    assert not failures, (
        "model discovery could not import these modules, so every model they register "
        f"is missing from the catalog: {failures}"
    )
