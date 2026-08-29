"""Regression test for the audit-15-F2 cleanup.

The 4 services that ``audit_services`` used to call optional —
``IManifestService``, ``IErrorHandlingService``,
``IModelManagementService``, ``IModelCompilationService``,
``IProfilingService`` — were registered in ``bootstrap.build_container``
but never resolved by any consumer in the training path. They were
removed to keep the DI container honest. If a future feature needs
them, register them back **with a corresponding ``container.resolve()``
call** so the wiring is testable end-to-end (CLAUDE.md pitfall #9).
"""

from __future__ import annotations

import inspect
from pathlib import Path


BANNED_REGISTRATIONS = (
    "register_service(IManifestService",
    "register_service(IErrorHandlingService",
    "register_service(IModelManagementService",
    "register_service(IModelCompilationService",
    "register_service(IProfilingService",
)


def _bootstrap_source() -> str:
    """Read bootstrap.py via its imported module path.

    The old hardcoded ``src/bootstrap.py`` went stale in the src→mriforge
    refactor and this guard silently failed on FileNotFoundError ever since
    (repaired 2026-07-01 — same fix as the sibling infer-kwargs guard).
    """
    import mriforge.bootstrap as bootstrap_mod

    return Path(bootstrap_mod.__file__).read_text(encoding="utf-8")


def test_bootstrap_does_not_register_unused_services() -> None:
    bootstrap_src = _bootstrap_source()
    leaked = [needle for needle in BANNED_REGISTRATIONS if needle in bootstrap_src]
    assert not leaked, (
        f"bootstrap.py re-introduced unused service registration(s): {leaked}. "
        "Audit 15 F2 found these had no consumer in the training path. If a "
        "real consumer now exists, surface the resolution at the call site "
        "(container.resolve(IService)) — don't leave a registered-but-never-"
        "resolved smell behind."
    )


def test_data_availability_check_has_no_silent_importerror_arm() -> None:
    """2026-07-01 cleanup: ``_validate_data_availability_at_startup`` carried a
    vestigial ``try: pass`` (from a removed import) and a dead
    ``except ImportError: skip validation`` arm — no import remained inside
    the try, and silently skipping validation is pitfall #9. Neither may
    return."""
    import mriforge.bootstrap as bootstrap_mod

    src = inspect.getsource(bootstrap_mod._validate_data_availability_at_startup)
    assert "except ImportError" not in src, (
        "silent ImportError-skip arm reappeared in the data-availability check"
    )
    assert "try:\n        pass" not in src, "vestigial `try: pass` reappeared"
