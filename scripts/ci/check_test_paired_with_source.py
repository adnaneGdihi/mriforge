"""Pre-commit / PR gate: fail if a src/spectramr/**.py change has no paired tests/** change.

Usage:
    python scripts/ci/check_test_paired_with_source.py                    # staged (pre-commit)
    python scripts/ci/check_test_paired_with_source.py --base A --head B  # commit range (PR gate)

Exit codes:
    0 — no source change, or every source change is accompanied by a tests/ change.
    1 — src/spectramr/ modules changed with NO tests/ file changed.

The PR gate MUST pass --base/--head. In CI nothing is staged, so the default staged
mode would report success without inspecting the pull request at all.

Wired into .pre-commit-config.yaml as a local hook (stages: [commit]) and into
.github/workflows/pr-required.yml as the `guards` job.
See CLAUDE.md "Source <-> test pairing (hard rule)" for the rationale.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def staged_files() -> list[str]:
    """Files staged for the next commit (Added/Copied/Modified)."""
    return _git("diff", "--cached", "--name-only", "--diff-filter=ACM")


def range_files(base: str, head: str) -> list[str]:
    """Files the PR adds or modifies, three-dot: head relative to the merge base.

    Two-dot would also report files that changed on the base branch after this branch
    forked, blaming the PR for someone else's commit.
    """
    return _git("diff", "--name-only", "--diff-filter=ACM", f"{base}...{head}")


def changed_sources(files: list[str]) -> list[str]:
    """Source modules whose change demands a test. __init__.py is re-exports only."""
    return [
        f
        for f in files
        if f.startswith("src/spectramr/") and f.endswith(".py") and not f.endswith("__init__.py")
    ]


def changed_tests(files: list[str]) -> list[str]:
    return [f for f in files if f.startswith("tests/") and f.endswith(".py")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Source <-> test pairing gate.")
    parser.add_argument("--base", help="Base commit of the range. Requires --head.")
    parser.add_argument("--head", help="Head commit of the range. Requires --base.")
    args = parser.parse_args(argv)

    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be given together")

    files = range_files(args.base, args.head) if args.base else staged_files()

    src_changed = changed_sources(files)
    test_changed = changed_tests(files)

    if src_changed and not test_changed:
        print(
            "ERROR: source file(s) changed without a paired test change.",
            file=sys.stderr,
        )
        for f in src_changed:
            print(f"  modified: {f}", file=sys.stderr)
        print(
            "\nAdd or extend a test under tests/ in the same change.\n"
            "See CLAUDE.md 'Source <-> test pairing (hard rule)' for details.\n"
            "Mapping convention: src/spectramr/<area>/<mod>.py"
            " <-> tests/unit/<area>/test_<mod>.py",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
