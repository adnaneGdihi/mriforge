"""Contract tests for ``scripts/ci/check_no_stale_package_name.py``.

Non-negotiable 15: a gate is only a gate for the violation shape it has been
watched to fail on. The *rule* this runner drives already plants its own needle
shapes; what is new here, and therefore what is planted below, is the runner's
own surface -- its three-valued exit code and the hook declaration that decides
whether it runs at all.

The exit codes are planted at the **call site** (``main``), never only in a
helper: a helper-only pin scores green on a ``main`` that computes the right
answer and then returns the wrong code, which is exactly a detector that never
fires.

The last test pins ``always_run`` in ``.pre-commit-config.yaml``. That is not
bookkeeping. Drop that one key and the hook silently becomes path-scoped, which
restores the precise blindness this guard exists to close: during the rename a
clean three-way merge carried two stale import lines into a file no path-level
check listed, because git routes hunks by path and never reads their contents.
A green suite with a path-scoped hook would look identical to a green suite with
a working one.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_no_stale_package_name.py"
CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
HOOK_ID = "no-stale-package-name"


def _load():
    """Load by path -- ``tests/unit/scripts`` shadows the root ``scripts``."""
    spec = importlib.util.spec_from_file_location("_stale_name_runner", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


runner = _load()

#: The rule owner, imported rather than re-spelled (non-negotiable 17). Note it
#: is imported for its *vocabulary*; this file never writes a retired name
#: literally, or the guard would report this file as an offender.
sys.path.insert(0, str(REPO_ROOT))
from tests.architecture import test_no_stale_package_name as owner  # noqa: E402


def test_clean_tree_exits_zero() -> None:
    """The state the tree is actually in, asserted rather than assumed."""
    assert runner.main() == 0


def test_offenders_exit_one_and_are_named(monkeypatch, capsys) -> None:
    """Planted at the call site: a finding must be 1, and must name the file."""
    monkeypatch.setattr(owner, "_offending_files", lambda: ["src/spectramr/x.py"])
    assert runner.main() == 1
    assert "src/spectramr/x.py" in capsys.readouterr().err


def test_import_failure_exits_two_not_one(monkeypatch, capsys) -> None:
    """A broken environment must not be reportable as a finding, or as clean.

    Exit 1 here would send someone hunting a file that is fine; exit 0 would be
    a silent pass. Absent is a state to report, never one to infer.
    """
    monkeypatch.delitem(sys.modules, owner.__name__, raising=False)
    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name.startswith("tests.architecture.test_no_stale"):
            raise ModuleNotFoundError("No module named 'pytest'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert runner.main() == 2
    assert "environment fault" in capsys.readouterr().err


def test_runner_delegates_to_the_owners_predicate() -> None:
    """Non-vacuity: the runner must reach the real rule, not a stub of it.

    A retired spelling is built from the owner's own fragments, so this test
    cannot drift from the vocabulary it checks -- and cannot match itself.
    """
    needle = owner.NEEDLES[0]
    assert owner._is_offending_text(f"import {needle}\n") is True
    assert owner._is_offending_text("import spectramr\n") is False


def test_the_hook_is_declared_always_run() -> None:
    """The property the hook was asked for, pinned where it is declared."""
    hooks = [
        h
        for repo in yaml.safe_load(CONFIG.read_text())["repos"]
        if repo["repo"] == "local"
        for h in repo["hooks"]
        if h["id"] == HOOK_ID
    ]
    assert len(hooks) == 1, f"expected exactly one {HOOK_ID} hook, found {len(hooks)}"
    hook = hooks[0]
    assert hook["always_run"] is True, "always_run dropped: hook is now path-scoped"
    assert hook["pass_filenames"] is False
    assert "files" not in hook, "a files: filter re-narrows an always_run hook"
    assert "pytest" in hook["additional_dependencies"], (
        "the rule owner imports pytest at module scope; without this the hook "
        "exits 2 in any environment that installs pre-commit alone"
    )
