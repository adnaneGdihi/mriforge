"""Regression test pinning the audit-14-F12 wiring.

Pre-fix audit (TODO/audit/14_physics_adapters_di_validation.md F12)
reported that ``service_audit.verify_startup_services`` existed but was
never invoked from ``spectramr.bootstrap.build_container`` — meaning DI
registration drift never failed loud at startup.

The call has since landed in ``src/bootstrap.py`` (Phase 4.5,
``fail_on_missing=True``). A future refactor that quietly removes the
call would put the codebase back into CLAUDE.md pitfall #9 territory
("registered-but-never-resolved" services with no startup signal).
This test pins the import + call site so the regression is loud.
"""

from __future__ import annotations

from pathlib import Path

import spectramr

# Derived from the package, not from a path literal. The 2026-05 refactor
# moved the tree to ``src/spectramr/`` and every hardcoded ``parents[N] / "src"``
# silently started pointing at a directory that does not exist -- 5 of cluster
# job 8004252's failures, all reading as FileNotFoundError rather than as the
# stale constant they were. ``spectramr.__file__`` cannot go stale on a move.
_PKG = Path(spectramr.__file__).resolve().parent


def test_bootstrap_calls_verify_startup_services() -> None:
    bootstrap_src = (
        _PKG / "bootstrap.py"
    ).read_text()

    assert (
        "from spectramr.infrastructure.di.service_audit import verify_startup_services"
        in bootstrap_src
    ), (
        "src/spectramr/bootstrap.py no longer imports verify_startup_services. "
        "The startup DI audit must run before training begins — "
        "see TODO/audit/14_physics_adapters_di_validation.md F12."
    )

    assert "verify_startup_services(container" in bootstrap_src, (
        "src/spectramr/bootstrap.py no longer invokes verify_startup_services. "
        "Without this call, services registered in build_container() that "
        "fail to resolve will only surface mid-training. Restore the "
        "call (with fail_on_missing=True) in PHASE 4.5."
    )

    assert "fail_on_missing=True" in bootstrap_src, (
        "verify_startup_services must be invoked with fail_on_missing=True "
        "so DI drift fails loud at startup, per CLAUDE.md pitfall #9."
    )
