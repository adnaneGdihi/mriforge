"""A ``pip install`` in a workflow must not carry an UPPER version bound.

Issue #839. The ``security`` job pinned ``setuptools>=78.1.1,<82``, the ceiling
copied from torch's own ``setuptools<82``. torch 2.13.0 dropped that ceiling, so
the copy became the only thing holding setuptools at 81.0.0 -- and when
PYSEC-2026-3447 landed (fixed in 83.0.0) the blocking job failed on every PR with
no version able to satisfy it.

The bound was correct when written. It went stale silently, and stayed invisible
because nothing had run since ~2026-07-13 (#831). **A bound copied from a
dependency's metadata is a snapshot, not a link.**

Lower bounds are fine and are how a CVE floor is expressed. An upper bound in CI
is the failure mode: it cannot track upstream, and it converts a fixable advisory
into an unsatisfiable gate -- the same "gate that can never go green" this repo
already rewrote ``lint-diff`` to avoid.

Let the real dependency resolver own ceilings. If a package genuinely constrains
another, pip enforces it loudly at install time; restating it here only adds a
second copy that can rot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROOTS = (
    _REPO_ROOT / ".github" / "workflows",
    _REPO_ROOT / ".github" / "actions",
)

# `pip install ...` up to end of line. Covers `python -m pip`, `--upgrade`, and
# quoted specs.
_PIP_INSTALL = re.compile(r"(?:python -m )?pip install [^\n]*")

# An upper bound: `<`, `<=`, or an exact `==` on a package spec. The operator must
# follow a package-name character OR A COMMA, and precede a digit, so a shell
# redirect (`< file`, `2>&1`) cannot false-positive.
#
# The comma is load-bearing and was missing in the first draft: the spec that
# caused #839 is `setuptools>=78.1.1,<82`, where the ceiling follows the `,` of a
# compound spec, not a name character. Without it this guard sailed past the exact
# string it was written to catch -- caught only because
# `test_upper_bound_regex_actually_matches_one` asserts against that literal.
_UPPER_BOUND = re.compile(r"[A-Za-z0-9_.,\-]\s*(?:<=?|==)\s*\d")


def _install_lines() -> list[tuple[Path, int, str]]:
    """Every `pip install` invocation across the workflow tree."""
    out: list[tuple[Path, int, str]] = []
    for root in _ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.yml")) + sorted(root.rglob("*.yaml")):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                for m in _PIP_INSTALL.finditer(line):
                    out.append((path, i, m.group(0)))
    return out


def test_parser_finds_pip_installs() -> None:
    """Positive control.

    Without this, a regex that matches nothing makes every assertion below pass
    vacuously -- and a green suite would then be evidence of a broken parser
    rather than a clean tree.
    """
    found = _install_lines()
    assert len(found) >= 3, f"expected several pip installs, parsed {len(found)}"


def test_upper_bound_regex_actually_matches_one() -> None:
    """Second positive control: the *bound* detector, not just the line finder.

    Asserted against the exact string that caused #839. A detector that silently
    stopped matching would make `test_no_upper_bound_pins` pass forever.
    """
    assert _UPPER_BOUND.search("pip install --upgrade 'setuptools>=78.1.1,<82'")
    assert _UPPER_BOUND.search("pip install 'foo==1.2.3'")
    assert not _UPPER_BOUND.search("pip install --upgrade 'setuptools>=83.0.0'")
    assert not _UPPER_BOUND.search("pip install pip-audit")


@pytest.mark.parametrize(
    ("path", "lineno", "cmd"),
    _install_lines(),
    ids=[f"{p.name}:{n}" for p, n, _ in _install_lines()],
)
def test_no_upper_bound_pins(path: Path, lineno: int, cmd: str) -> None:
    """No `pip install` in CI may cap a package's version. Issue #839."""
    assert not _UPPER_BOUND.search(cmd), (
        f"{path.relative_to(_REPO_ROOT)}:{lineno} caps a dependency: {cmd!r}\n"
        "An upper bound in CI cannot track upstream. When the real constraint "
        "moves, this copy silently keeps the old ceiling, and a CVE fixed above "
        "it makes the job unsatisfiable (#839: setuptools<82 vs PYSEC-2026-3447 "
        "fixed in 83.0.0). Express CVE floors with `>=`; let pip own ceilings."
    )
