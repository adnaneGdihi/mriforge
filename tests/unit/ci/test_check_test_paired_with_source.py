"""Tests for the source<->test pairing gate (CLAUDE.md non-negotiable #10).

The gate's PR mode is the load-bearing case: in CI nothing is staged, so a
staged-only gate would pass vacuously on every pull request.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_test_paired_with_source.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_pairing_gate", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_script()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo with one seed commit, cwd'd into."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "README.md").write_text("seed\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-qm", "seed")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit(repo: Path, *rel_paths: str) -> str:
    for rel in rel_paths:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n")
        _git(repo, "add", rel)
    _git(repo, "commit", "-qm", f"add {' '.join(rel_paths)}")
    return _git(repo, "rev-parse", "HEAD")


def test_source_without_test_fails(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, "src/spectramr/core/thing.py")
    assert gate.main(["--base", base, "--head", head]) == 1


def test_source_with_test_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, "src/spectramr/core/thing.py", "tests/unit/core/test_thing.py")
    assert gate.main(["--base", base, "--head", head]) == 0


def test_init_only_change_passes(repo: Path) -> None:
    """__init__.py is re-exports; it carries no behavior to test."""
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, "src/spectramr/core/__init__.py")
    assert gate.main(["--base", base, "--head", head]) == 0


def test_non_python_change_passes(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    head = _commit(repo, "docs/note.md")
    assert gate.main(["--base", base, "--head", head]) == 0


def test_staged_mode_still_works_with_nothing_staged(repo: Path) -> None:
    """The pre-commit hook calls the script with no args. Must not regress."""
    assert not _git(repo, "diff", "--cached", "--name-only")
    assert gate.main([]) == 0


def test_staged_mode_catches_unpaired_source(repo: Path) -> None:
    path = repo / "src" / "spectramr" / "core" / "thing.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n")
    _git(repo, "add", "src/spectramr/core/thing.py")
    assert gate.main([]) == 1


def test_base_without_head_is_a_usage_error(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(SystemExit) as excinfo:
        gate.main(["--base", head])
    assert excinfo.value.code == 2
