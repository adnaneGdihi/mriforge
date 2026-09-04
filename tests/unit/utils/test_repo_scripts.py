"""Planted violations for :mod:`tests.utils.repo_scripts`.

Non-negotiable 15: a gate is only a gate for the violation shape it has been
watched failing on. This helper decides whether a missing ``scripts/`` file is a
publication boundary (skip) or a defect (fail), and **both** of its wrong answers
are silent -- skipping in a private checkout hides a deletion, failing in the
export re-breaks collection for the whole suite. So each branch of the table in
the helper's docstring is planted here against a synthetic tree, and
``test_live_tree_is_not_an_export`` pins the answer for the tree the suite is
actually running in, which is the only plant that catches the helper degrading
into "always skip".

The functions take their ``root``, so the export branch is reachable from the
private checkout. Driving it with the live tree instead would leave that branch
untested in the one tree that can run these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.repo_scripts import (
    ALLOWLIST_REL,
    REPO_ROOT,
    is_public_export,
    load_script_module,
    require_script,
    selects,
    unselected_scripts,
)

_ALLOWLIST = """\
# a comment line, selecting nothing
scripts/release/
scripts/ci/shipped.py          # a real entry with a trailing comment

"""


def _tree(tmp_path: Path, *extra: str) -> Path:
    """A tree that ships exactly the allowlisted set, plus ``extra`` files."""
    root = tmp_path
    (root / "scripts" / "release").mkdir(parents=True)
    (root / "scripts" / "ci").mkdir(parents=True)
    (root / ALLOWLIST_REL).write_text(_ALLOWLIST, encoding="utf-8")
    (root / "scripts" / "ci" / "shipped.py").write_text("VALUE = 1\n", encoding="utf-8")
    for rel in extra:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 2\n", encoding="utf-8")
    return root


# --- selects(): the three line shapes the allowlist actually mixes -----------


@pytest.mark.parametrize(
    ("entry", "rel", "expected"),
    [
        ("scripts/release/", "scripts/release/deep/nested.py", True),  # dir prefix
        ("scripts/release/", "scripts/releaser.py", False),  # prefix is not a substring
        ("scripts/ci/x.py", "scripts/ci/x.py", True),  # exact file
        ("scripts/ci/x.py", "scripts/ci/x.pyc", False),  # exact means exact
        ("scripts/ci/x.py  # why", "scripts/ci/x.py", True),  # trailing comment
        ("# scripts/ci/x.py", "scripts/ci/x.py", False),  # commented-OUT entry
        ("   ", "scripts/ci/x.py", False),  # blank
    ],
)
def test_selects_line_shapes(entry: str, rel: str, expected: bool) -> None:
    assert selects(entry, rel) is expected


# --- the export verdict -----------------------------------------------------


def test_tree_shipping_only_allowlisted_files_is_an_export(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert unselected_scripts(root) == ()
    assert is_public_export(root) is True


def test_one_unselected_file_makes_it_a_private_tree(tmp_path: Path) -> None:
    """The plant: a single unshipped script must flip the verdict."""
    root = _tree(tmp_path, "scripts/ci/private_only.py")
    assert unselected_scripts(root) == ("scripts/ci/private_only.py",)
    assert is_public_export(root) is False


def test_missing_allowlist_is_not_an_export(tmp_path: Path) -> None:
    """Absent is reported, not inferred: an unrecognisable tree earns no skip."""
    root = _tree(tmp_path)
    (root / ALLOWLIST_REL).unlink()
    assert is_public_export(root) is False


def test_bytecode_does_not_make_a_tree_private(tmp_path: Path) -> None:
    """``__pycache__`` is whoever imported last, never evidence about the tree."""
    root = _tree(tmp_path, "scripts/ci/__pycache__/shipped.cpython-312.pyc")
    assert is_public_export(root) is True


def test_live_tree_is_not_an_export() -> None:
    """Non-vacuity, in the tree the suite is running in.

    If this ever returns True in the private checkout, every ``require_script``
    call site turns into an unconditional skip and the suite goes quietly green.
    """
    assert is_public_export(REPO_ROOT) is False
    assert len(unselected_scripts(REPO_ROOT)) > 100


# --- require_script(): both wrong answers are planted -----------------------


def test_present_script_is_returned(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert require_script("scripts/ci/shipped.py", root) == root / "scripts/ci/shipped.py"


def test_absent_in_export_skips_and_names_the_reason(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(pytest.skip.Exception, match="does not ship in the public export"):
        require_script("scripts/ci/absent.py", root)


def test_absent_in_a_private_tree_fails_rather_than_skipping(tmp_path: Path) -> None:
    """The masking plant: a deletion must NOT be mistaken for a boundary."""
    root = _tree(tmp_path, "scripts/ci/private_only.py")
    with pytest.raises(pytest.fail.Exception, match="NOT the public export"):
        require_script("scripts/ci/absent.py", root)


def test_absent_with_no_allowlist_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / ALLOWLIST_REL).unlink()
    with pytest.raises(pytest.fail.Exception, match="NOT the public export"):
        require_script("scripts/ci/absent.py", root)


# --- load_script_module() ---------------------------------------------------


def test_load_script_module_executes_and_registers(tmp_path: Path) -> None:
    import sys

    root = _tree(tmp_path)
    mod = load_script_module("scripts/ci/shipped.py", "shipped_under_test", root)
    assert mod.VALUE == 1
    assert sys.modules["shipped_under_test"] is mod
    del sys.modules["shipped_under_test"]


def test_load_script_module_inherits_the_skip(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(pytest.skip.Exception):
        load_script_module("scripts/ci/absent.py", "absent_under_test", root)
