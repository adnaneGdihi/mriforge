"""``scripts/ci/lint_changed_lines.py`` gates NEW debt, not inherited debt.

The gate it replaces ran whole-file ruff on every touched file. On a tree with 13,225
pre-existing findings that made it unsatisfiable: every PR that touched a legacy module
went red for debt it did not write, so ``required`` was red on all ten PRs of the
sim2rank stack while the four lanes that actually test correctness were green. A check
that can never pass is not a gate -- it trains people to merge red.

These tests pin the two halves of the contract:
  * a finding on a line the PR did not touch must NOT fail it, and
  * a finding on a line the PR DID add must.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "lint_changed_lines.py"

# N806: `B0` is a non-lowercase local. It is also the correct symbol for the main
# magnetic field, which is why 146 of these sit in the physics tree and why nobody is
# going to rename them.
_DIRTY = "def f():\n    B0 = 3.0\n    return B0\n"
_CLEAN = "def g():\n    return 1\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "src").mkdir(parents=True)
    _git(r.parent, "init", "-q", "-b", "main", str(r))
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")
    # ruff needs a config, or it picks up the repo's own pyproject from a parent dir.
    (r / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (r / "ruff.toml").write_text('line-length = 100\nlint.select = ["E", "F", "N"]\n')
    return r


def _run(repo: Path, base: str, head: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--base", base, "--head", head],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_inherited_debt_on_an_untouched_line_does_not_fail_the_pr(repo: Path) -> None:
    """THE case. Legacy file is already dirty; the PR appends a clean function."""
    (repo / "src" / "legacy.py").write_text(_DIRTY)
    base = _commit(repo, "pre-existing debt")

    (repo / "src" / "legacy.py").write_text(_DIRTY + "\n\n" + _CLEAN)
    head = _commit(repo, "append something clean")

    res = _run(repo, base, head)
    assert res.returncode == 0, f"inherited N806 must not fail the PR:\n{res.stdout}"


def test_a_violation_on_a_line_the_pr_added_fails(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text(_CLEAN)
    base = _commit(repo, "clean file")

    (repo / "src" / "legacy.py").write_text(_CLEAN + "\n\n" + _DIRTY)
    head = _commit(repo, "add new debt")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "N806" in res.stdout
    assert "line added by this PR" in res.stdout


def test_a_new_file_is_gated_end_to_end(repo: Path) -> None:
    """An added file has no inherited debt, so every finding in it is the PR's."""
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "brand_new.py").write_text(_DIRTY)
    head = _commit(repo, "add a dirty new file")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "new file" in res.stdout


def test_a_new_file_must_also_be_formatted(repo: Path) -> None:
    (repo / "src" / "keep.py").write_text(_CLEAN)
    base = _commit(repo, "seed")

    (repo / "src" / "ugly.py").write_text("def h():\n    return   [1,2,   3]\n")
    head = _commit(repo, "add an unformatted new file")

    res = _run(repo, base, head)
    assert res.returncode == 1
    assert "not ruff-formatted" in res.stdout


def test_a_modified_file_is_not_conscripted_into_a_reformat(repo: Path) -> None:
    """Reformatting a legacy file to satisfy a two-line diff buries the change."""
    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n")
    base = _commit(repo, "badly formatted legacy file")

    (repo / "src" / "legacy.py").write_text("def h():\n    return   [1,2,   3]\n\n\n" + _CLEAN)
    head = _commit(repo, "append a clean function")

    res = _run(repo, base, head)
    assert res.returncode == 0, res.stdout


def test_a_pr_touching_no_python_passes(repo: Path) -> None:
    (repo / "README.md").write_text("hi\n")
    base = _commit(repo, "seed")
    (repo / "README.md").write_text("hi there\n")
    head = _commit(repo, "docs only")

    res = _run(repo, base, head)
    assert res.returncode == 0
    assert "No changed Python files" in res.stdout
