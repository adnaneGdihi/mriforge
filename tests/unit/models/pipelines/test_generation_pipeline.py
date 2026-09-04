"""Characterization tests for the unwired 3D-generation pipeline module.

``spectramr.models.pipelines.generation_pipeline`` used to be *broken* as well as
unwired: ``from ....services.device_policy import ...`` reached above the
``spectramr`` top-level package (a ``src -> spectramr`` refactor leftover) and
raised ``ImportError`` on import. Its only consumer,
``domain.services.generation_3d_orchestration_service``, catches that with
``except ImportError: generation_pipeline_module = None`` — so a genuinely broken
module was indistinguishable from an absent optional dependency, in every
environment, forever (non-negotiable 18).

The imports are absolute now. That makes the module importable and its remaining
deadness *visible*; it wires nothing. These tests pin exactly that state — the
import works, the chain is still unreachable — so neither half can regress
silently and the keep-or-wire decision stays deliberate. See
``TODO/backlog_dead_code_pipelines_audit_2026_06_12.md``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_MODULE = "spectramr.models.pipelines.generation_pipeline"


def _source_path() -> Path:
    import spectramr

    pkg_root = Path(next(iter(spectramr.__path__)))
    return pkg_root / "models" / "pipelines" / "generation_pipeline.py"


def test_module_imports() -> None:
    """The beyond-top-level import is repaired: the module loads."""
    assert importlib.import_module(_MODULE) is not None


def test_no_beyond_top_level_relative_imports() -> None:
    """The specific defect shape, not just its symptom.

    ``....x`` from ``spectramr/models/pipelines/`` walks off the top of the
    package. Asserting on importability alone would pass if someone reintroduced
    one inside a function body, where import-time never reaches it.
    """
    text = _source_path().read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if line.lstrip().startswith("from ....") or line.lstrip().startswith("import ....")
    ]
    assert not offenders, f"beyond-top-level relative imports reintroduced: {offenders}"


def test_consumer_no_longer_sees_it_as_an_absent_optional_dependency() -> None:
    """The ``except ImportError`` guard must not be what resolves this module.

    This is the assertion that would have caught the original defect: the guard
    silently bound ``None``, so the chain reported "optional dependency missing"
    when the truth was "our own module is broken".
    """
    service = importlib.import_module(
        "spectramr.domain.services.generation_3d_orchestration_service"
    )
    assert service.generation_pipeline_module is not None, (
        "the consumer still binds None — a broken in-tree module is being "
        "misreported as an absent optional dependency (non-negotiable 18)"
    )


def test_module_still_flags_itself_as_a_removal_candidate() -> None:
    """Importable is not wired. The banner must keep saying so."""
    text = _source_path().read_text(encoding="utf-8")
    assert "UNWIRED" in text
    assert "backlog_dead_code_pipelines" in text


def test_chain_is_still_unreachable_from_production() -> None:
    """Pin the deadness itself, so 'fixed the import' is never read as 'wired'.

    If this fails because a real caller appeared, that is the keep-or-wire
    decision being made — update this test as part of making it.
    """
    import subprocess

    root = Path(next(iter(importlib.import_module("spectramr").__path__)))
    hits = subprocess.run(
        [
            "grep",
            "-rln",
            "generation_3d_orchestration_service",
            "--include=*.py",
            str(root),
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    callers = [
        h
        for h in hits
        if not h.endswith("generation_3d_orchestration_service.py")
        and not h.endswith("generation_pipeline.py")
        and not h.endswith("entities_3d.py")  # docstring example only
    ]
    assert not callers, (
        f"the 3D-generation chain gained production callers: {callers}. "
        "It is no longer dead — re-pin these characterization tests."
    )
