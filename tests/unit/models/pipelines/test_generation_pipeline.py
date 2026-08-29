"""Characterization tests for the dead 3D-generation pipeline module.

``mriforge.models.pipelines.generation_pipeline`` is broken dead code: its
``from ....services.device_policy import ...`` reaches above the ``mriforge``
top-level package (a leftover from the ``src -> mriforge`` refactor), so the
module raises ``ImportError`` on import. Its only consumer
(``domain.services.generation_3d_orchestration_service``) swallows that via a
``try/except ImportError`` guard, leaving the whole 3D-generation chain
unreachable.

These tests pin the known-dead state and the in-file removal-candidate banner
so the module cannot be silently "half-fixed" (import repaired while still
unwired) without forcing a deliberate keep-or-remove decision. See
``TODO/backlog_dead_code_pipelines_audit_2026_06_12.md``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_MODULE = "mriforge.models.pipelines.generation_pipeline"


def _source_path() -> Path:
    import mriforge

    pkg_root = Path(next(iter(mriforge.__path__)))
    return pkg_root / "models" / "pipelines" / "generation_pipeline.py"


def test_module_is_currently_unimportable() -> None:
    """The module is dead: it cannot be imported (beyond-top-level import)."""
    with pytest.raises(ImportError):
        importlib.import_module(_MODULE)


def test_module_carries_removal_candidate_banner() -> None:
    """The source flags itself as a dead removal candidate for readers."""
    text = _source_path().read_text(encoding="utf-8")
    assert "DEAD CODE" in text
    assert "backlog_dead_code_pipelines" in text
