"""Regression test for the audit-17-F5 fix.

``src/infrastructure/coordination/hpo_coordinator.py`` used to import
``from spectramr.pipelines.hpo import subprocess_training_objective`` —
infrastructure reaching upward into pipelines, which is a
CLAUDE.md layer-direction violation. The factory is now injected by
the caller (application/use_cases/hpo_use_case.py); the coordinator
raises if the factory is not provided.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import spectramr

# Derived from the package, not from a path literal. The 2026-05 refactor
# moved the tree to ``src/spectramr/`` and every hardcoded ``parents[N] / "src"``
# silently started pointing at a directory that does not exist -- 5 of cluster
# job 8004252's failures, all reading as FileNotFoundError rather than as the
# stale constant they were. ``spectramr.__file__`` cannot go stale on a move.
_PKG = Path(spectramr.__file__).resolve().parent


def test_hpo_coordinator_does_not_import_from_pipelines() -> None:
    src = (
        _PKG
        / "infrastructure"
        / "coordination"
        / "hpo_coordinator.py"
    ).read_text()
    assert "from spectramr.pipelines" not in src, (
        "hpo_coordinator.py reintroduced an upward import from "
        "spectramr.pipelines. The objective factory must be injected via "
        "the `objective_factory=` parameter — see audit 17 F5."
    )


def test_hpo_coordinator_requires_objective_factory(tmp_path) -> None:
    from spectramr.infrastructure.coordination.hpo_coordinator import HPOCoordinator

    # Make a config_path that exists so we get past the FileNotFoundError
    # check and hit the actual objective_factory guard.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model:\n  model_type: dummy_unet\n")

    coordinator = HPOCoordinator()
    with pytest.raises(ValueError, match="objective_factory"):
        coordinator.execute_hpo(
            model_types=["dummy_unet"],
            input_lr_dir=str(tmp_path),
            input_hr_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            config_path=str(cfg),
            # Deliberately omit objective_factory.
        )


def test_hpo_use_case_injects_subprocess_factory() -> None:
    """The application layer is the canonical injection point for the
    pipeline-side factory."""
    src = (
        _PKG
        / "application"
        / "use_cases"
        / "hpo_use_case.py"
    ).read_text()
    assert "from spectramr.pipelines.hpo import subprocess_training_objective" in src
    assert "objective_factory=subprocess_training_objective" in src
