#!/usr/bin/env python3
"""Ruff, scoped to the lines a PR actually changed.

Why not just ``ruff check <changed files>``
-------------------------------------------
Because ``dev`` carries **13,225** pre-existing ruff findings across **2,526**
unformatted files. A whole-file gate therefore fails on debt the PR did not write:
touch one line of a physics module and you inherit all of its ``N806`` findings --
and ``N806`` (non-lowercase-variable-in-function) fires on ``B0``, ``T1``, ``M0``,
``S``, ``F``, which are the *correct* symbols in MRI. The gate was unsatisfiable for
any PR touching a legacy file, which is the same as no gate at all: a check that can
never be green teaches everyone to merge red, and then a real failure walks straight
through.

The workflow already says its intent is to lint "what GitHub's Files changed tab
shows". This implements exactly that, at line granularity instead of file granularity.

The rules
---------
* **Added file** -> must be clean end to end (``ruff check`` AND ``ruff format``).
  New code has no inherited debt and no excuse.
* **Modified file** -> only findings ON LINES THE PR ADDED are errors. Format is not
  checked: reformatting a legacy file to satisfy a two-line diff is pure churn, and it
  buries the actual change in a 400-line whitespace commit.

So a PR is free to leave old debt alone, and is not free to add new debt. Cleaning a
legacy file remains welcome -- it just is not conscripted.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

#: ``@@ -old,+new @@`` -- we only care about the post-image (the ``+`` side), because
#: that is what the PR is proposing to land.
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

_ROOTS = ("src/", "tests/")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _in_scope(path: str) -> bool:
    return path.endswith(".py") and path.startswith(_ROOTS)


def changed_files(base: str, head: str) -> tuple[list[str], list[str]]:
    """``(added, modified)`` Python paths, three-dot (i.e. relative to the merge base).

    Two-dot would also blame this PR for commits that landed on the base branch after
    the branch forked.

    Scoping happens in Python, NOT via a ``src/**/*.py`` pathspec: git's wildmatch wants
    an intermediate directory for ``**``, so that pathspec matches
    ``src/spectramr/models/x.py`` but silently MISSES ``src/x.py``. The whole-file gate
    this replaces had exactly that hole -- it only ever worked because everything in
    this repo happens to sit one level down, under ``src/spectramr/``.
    """
    out = _git("diff", "--name-status", "--diff-filter=ACM", f"{base}...{head}")
    added: list[str] = []
    modified: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        if not _in_scope(path):
            continue
        if status.startswith("A"):
            added.append(path)
        else:  # C (copied) counts as new content too, but git reports it rarely.
            modified.append(path)
    return added, modified


def added_lines(base: str, head: str, path: str) -> set[int]:
    """1-indexed line numbers this PR ADDS to ``path``, in the head revision.

    ``-U0`` so the hunk headers carry no context lines -- a context line is code the
    PR did not touch, and blaming the PR for a finding on it is the whole bug.
    """
    out = _git("diff", "-U0", f"{base}...{head}", "--", path)
    lines: set[int] = set()
    for raw in out.splitlines():
        m = _HUNK.match(raw)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            lines.update(range(start, start + count))
    return lines


def ruff_findings(paths: list[str]) -> list[dict]:
    """``ruff check --output-format=json``. Exit code is ignored: findings are the signal."""
    if not paths:
        return []
    proc = subprocess.run(
        ["ruff", "check", "--output-format=json", "--", *paths],
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ruff failed to run: {proc.stderr.strip()}")
        return []
    return json.loads(proc.stdout)


def unformatted(paths: list[str]) -> list[str]:
    if not paths:
        return []
    proc = subprocess.run(
        ["ruff", "format", "--check", "--", *paths], capture_output=True, text=True
    )
    return [
        line.removeprefix("Would reformat: ").strip()
        for line in proc.stdout.splitlines()
        if line.startswith("Would reformat:")
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="merge-base side SHA")
    ap.add_argument("--head", required=True, help="PR head SHA")
    args = ap.parse_args()

    added, modified = changed_files(args.base, args.head)
    if not added and not modified:
        print("No changed Python files under src/ or tests/.")
        return 0

    print(f"{len(added)} added file(s), {len(modified)} modified file(s)")

    violations: list[str] = []

    root = Path.cwd().resolve()

    def _rel(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(root))
        except ValueError:
            return p

    # Added files: every finding counts. New code has no inherited debt.
    for f in ruff_findings(added):
        loc = f.get("location") or {}
        violations.append(
            f"{_rel(f['filename'])}:{loc.get('row', '?')}:{loc.get('column', '?')}: "
            f"{f.get('code')} {f.get('message')}  [new file]"
        )
    for path in unformatted(added):
        violations.append(f"{_rel(path)}: not ruff-formatted  [new file]")

    # Modified files: only findings on lines this PR added.
    scoped = {p: added_lines(args.base, args.head, p) for p in modified}
    for f in ruff_findings(modified):
        loc = f.get("location") or {}
        path = _rel(f["filename"])
        if loc.get("row") in scoped.get(path, ()):
            violations.append(
                f"{path}:{loc.get('row')}:{loc.get('column', '?')}: "
                f"{f.get('code')} {f.get('message')}  [line added by this PR]"
            )

    if violations:
        print(f"\n{len(violations)} ruff violation(s) introduced by this PR:\n")
        for v in violations:
            print(f"  {v}")
        print(
            "\nOnly NEW code is gated: added files must be clean end to end, and "
            "modified files only on the lines you added. Pre-existing findings "
            "elsewhere in a file you touched are NOT your problem."
        )
        return 1

    print("No ruff violations on lines this PR added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
