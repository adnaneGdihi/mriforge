#!/usr/bin/env python3
"""Fail the commit when a tracked file still carries the pre-rename package name.

This is the pre-commit *runner* for a rule that already has an owner. The
vocabulary (which spellings count as the old name), the waiver policy (which
historical paths are correct as written) and the predicate that decides an
offence all live in ``tests/architecture/test_no_stale_package_name.py`` and are
**imported** here rather than re-spelled -- non-negotiable #17. A second
spelling of ``NEEDLES`` would not announce itself: both would find most
offenders, both would pass their own tests, and the divergence would surface as
a file that shipped with a dead import rather than as an error.

Why a runner exists at all, when the rule is already a test: the guard has to
hold on **every** commit, and the suite that carries it does not run on every
commit. ``pre-commit`` drives this file with ``always_run: true``, so the check
is independent of which paths a given commit happens to touch -- which is the
shape that leaked. During the most recent package rename, dev added two
imports of the retired package to a file whose merge was clean and whose path
nothing flagged; only a whole-tree *content* scan found them.

This docstring deliberately does not spell the retired name. The rule is
content-based and does not exempt its own runner, so a literal here would make
this file an offender -- which is how the omission was discovered rather than
foreseen.

What this scans is the **index**, not the working tree, and that distinction
was measured rather than assumed. ``git grep`` with no ref reads the working
tree, but ``pre-commit`` stashes unstaged changes before running any hook, so
what the scan sees is exactly the content the commit would create. A violation
left unstaged is therefore *correctly* not reported -- it is not being
committed. The consequence worth knowing: this cannot be exercised by editing a
file and committing something else. A plant has to be staged, or it is silently
stashed away and the gate reports clean (observed, non-negotiable 15).

Scope is the whole tree, not the changed files. A stale name anywhere in a
tracked file fails the commit even when nothing in that file is being touched,
which is the point -- the leak that motivated this arrived in a file the commit
did not name.

Exit codes are three-valued on purpose:

    0  clean
    1  offending files found -- the finding this exists to report
    2  the rule owner could not be imported

2 is not 1. A provisioning failure that reported "no offenders" would be a
silent pass, and one that reported an offence would send someone hunting a file
that is fine. Absent is a state to report, never a state to infer.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: scripts/ci/<this>.py -> scripts/ -> repo root. Derived from ``__file__`` and
#: never from the cwd: a repo-rooted script invoked from elsewhere resolves its
#: root to "/" and then reports a vacuously clean tree.
REPO_ROOT = Path(__file__).resolve().parents[2]

RULE_OWNER = "tests/architecture/test_no_stale_package_name.py"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from tests.architecture.test_no_stale_package_name import (
            NEEDLES,
            _offending_files,
        )
    except ModuleNotFoundError as exc:
        # Name the missing package. The owner imports pytest at module scope
        # (for ``pytestmark`` and one ``pytest.fail``), so a hook environment
        # without it fails here rather than at the finding -- and the two must
        # not look alike.
        print(
            f"cannot import the rule owner ({RULE_OWNER}): {exc}\n"
            "This is an environment fault, not a finding. The pre-commit hook "
            "provisions the owner's imports via `additional_dependencies`; a "
            "direct invocation needs them on the interpreter running this file.",
            file=sys.stderr,
        )
        return 2

    offenders = _offending_files()
    if not offenders:
        print(f"stale package name: clean ({len(NEEDLES)} spellings checked)")
        return 0

    print(
        f"{len(offenders)} tracked file(s) still carry the pre-rename package "
        "name as an identifier.\n"
        "A stale dotted path does not raise on a machine where the old package "
        "is still installed -- it silently resolves to the old tree.\n",
        file=sys.stderr,
    )
    for path in offenders:
        print(f"  {path}", file=sys.stderr)
    print(
        f"\nRewrite them, then re-commit:\n"
        f"  python scripts/migrations/rename_package_identifier.py --apply\n"
        f"The waiver list for paths that are correct as written lives in "
        f"{RULE_OWNER}.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
