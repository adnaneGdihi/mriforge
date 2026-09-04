"""Whole-tree ratchet for ``F821`` (undefined name). The count may only fall.

Why this exists alongside ``lint_changed_lines.py``
---------------------------------------------------
The ``lint-diff`` job deliberately scopes ruff to the lines a PR *adds*, and that
decision is right: whole-file ruff on ``dev`` reports 13,225 findings over 2,526
files, so any PR touching a legacy module would go red for debt it did not write,
and a gate that can never be green teaches everyone to merge red.

But F821 is not a style rule -- an undefined name is a ``NameError`` waiting for
the branch that reaches it -- and a diff-scoped linter is STRUCTURALLY unable to
catch the way F821 actually appears here. Non-negotiable #19's example is exactly
this: commit ``73a81dd71`` rewrote ``print(...)`` -> ``logger.debug(...)`` without
adding the binding. A mechanical rewrite is locally well-formed everywhere it is
wrong, and it breaks lines nobody edited, so nothing in the PR's own diff is
newly-red. Same shape found while adding this gate: ``streaming_inference.py``
had its ``import nibabel`` removed for the data-SSOT rule while two ``nib.*``
call sites stayed (#1403).

So: whole tree, but ratcheted rather than clean -- 55 recorded, and the number
may only go down.

Three failure classes, kept apart on purpose
--------------------------------------------
1. **New/grown F821.** A ``(path, name)`` pair not recorded, or recorded with a
   smaller count. Counting matters for the same reason the LOC ratchet's does
   (non-negotiable #20): keying on identity alone means a second undefined
   ``logger`` beside a recorded one lands green forever.
2. **Fixed F821 still recorded.** A hard failure, not a courtesy: a stale entry
   pre-exempts the name if it comes back. Matches the sibling allow-lists.
3. **Unparseable files.** ruff reports these as ``invalid-syntax``, NOT F821, and
   one broken file produced 270 of them -- 6x the real signal. Folding them in
   would make the number meaningless. Worse, a file that does not parse has
   *unknown* F821 status: every AST gate in this repo does
   ``except SyntaxError: continue``, so it is silently exempt from all of them.
   Reported as its own class against its own list (non-negotiable #18: absent is
   a state to report, never one to infer).

Usage:
    python scripts/ci/check_f821_ratchet.py            # check
    python scripts/ci/check_f821_ratchet.py --write    # re-record (only to LOWER)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE_FILENAME = "f821.txt"
UNPARSEABLE_FILENAME = "f821_unparseable.txt"
_NAME_RE = re.compile(r"`([^`]+)`")


#: Dot-directories that hold no first-party source and must not be walked.
#: ``.git`` is enormous, and the tool caches are generated. Anything else --
#: ``.claude/`` and ``.agent/`` today -- carries tracked ``.py`` files and is
#: therefore in scope, so this is an explicit denylist rather than an allowlist:
#: a NEW dot-directory of ours joins the gate's reach by default instead of
#: silently sitting outside it.
_UNSCANNED_DOT_DIRS = frozenset(
    {".git", ".venv", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".idea", ".vscode", ".tox"}
)


def _scannable(path: Path) -> bool:
    """Whether a dot-directory holds first-party code worth linting."""
    return path.name not in _UNSCANNED_DOT_DIRS


def ruff_scan_roots(root: Path) -> list[str]:
    """The paths this gate hands to ruff — the SSOT for its reach.

    Public and used by ``tests/unit/ci/test_check_f821_ratchet.py`` rather than
    restated there. The test previously spelled ``["."]`` out for itself, so it
    measured a reach the gate no longer had: two owners for one question
    (non-negotiable 17), and the coverage assertion could not see a fix to the
    gate it was auditing.
    """
    return [
        ".",
        *(p.name for p in sorted(root.glob(".*")) if p.is_dir() and _scannable(p)),
    ]


def _run_ruff(root: Path) -> list[dict]:
    """Ruff's F821 diagnostics as JSON. A missing ruff RAISES rather than passing."""
    if shutil.which("ruff") is None:
        raise RuntimeError(
            "ruff is not on PATH. This gate needs it; skipping would report "
            "'no undefined names' when the truth is 'nothing was checked'."
        )
    # ``.`` alone is NOT the whole tree. Ruff skips dot-directories while
    # walking, so every tracked ``.py`` under ``.claude/`` was outside this
    # gate's reach -- an undefined name there could never turn it red. That is
    # not hypothetical: the 6 scripts added with the ``academic-prose`` skill
    # landed straight into the blind spot, and only
    # ``test_ruff_visits_every_tracked_python_file`` noticed. Naming the
    # directory explicitly makes ruff descend into it (verified: it reports
    # findings for ``.claude`` when named, none when only ``.`` is passed).
    #
    # A detector is only a detector for what it can SEE (non-negotiable 15), and
    # the reach was wrong in BOTH directions: too narrow here, and too wide over
    # ``external/`` (git submodules), which ``[tool.ruff] extend-exclude`` now
    # removes -- third-party code this repo neither owns nor can fix.
    proc = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--select",
            "F821",
            "--output-format",
            "json",
            *ruff_scan_roots(root),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"ruff produced no output (exit {proc.returncode}): {proc.stderr[:400]}")
    return json.loads(proc.stdout)


def split_diagnostics(
    diagnostics: list[dict], root: Path
) -> tuple[Counter[tuple[str, str]], set[str]]:
    """Split ruff's output into ``(F821 counts, unparseable files)``."""
    counts: Counter[tuple[str, str]] = Counter()
    unparseable: set[str] = set()
    for d in diagnostics:
        rel = Path(d["filename"]).resolve().relative_to(root.resolve()).as_posix()
        if d.get("code") == "F821":
            match = _NAME_RE.search(d.get("message", ""))
            counts[(rel, match.group(1) if match else "?")] += 1
        else:  # invalid-syntax and anything else ruff emits alongside
            unparseable.add(rel)
    return counts, unparseable


def read_baseline(path: Path) -> Counter[tuple[str, str]]:
    """Parse ``count<TAB>path<TAB>name`` rows; a missing file RAISES."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. An absent baseline would make every recorded "
            "undefined name read as NEW, or (worse, if defaulted to empty) make "
            "the gate pass while checking nothing."
        )
    counts: Counter[tuple[str, str]] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        count, rel, name = line.split("\t")
        counts[(rel, name)] = int(count)
    return counts


def read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def compare(
    found: Counter[tuple[str, str]],
    baseline: Counter[tuple[str, str]],
    unparseable: set[str],
    known_unparseable: set[str],
    root: Path | None = None,
    absent_files_ok: bool = False,
) -> tuple[list[str], list[str]]:
    """Every way the tree may not have moved. Empty problems means the ratchet held.

    ``absent_files_ok`` is for a tree that deliberately contains fewer files than
    the baseline was recorded on -- the public export, which drops
    ``scripts/sim2rank/`` and ``scripts/preprocessing/`` wholesale. It downgrades
    ONLY the rows whose file is missing from ``root``, and only in the
    count-went-down direction; a row whose file is PRESENT still fails exactly as
    before, so the tree that recorded the baseline keeps the full check. Absences
    are returned and printed rather than dropped: a row that silently stops
    applying is how a ratchet turns into an empty ceremony. Do NOT reach for
    ``--write`` here -- regenerating accepts whatever the shipped files currently
    carry, which would baseline real defects (non-negotiable 20).
    """
    problems: list[str] = []
    absent: list[str] = []

    def _is_absent(rel: str) -> bool:
        return absent_files_ok and root is not None and not (root / rel).exists()

    for key in sorted(found.keys() | baseline.keys()):
        rel, name = key
        now, was = found[key], baseline[key]
        if now > was:
            problems.append(
                f"{rel}: undefined name `{name}` x{now} (recorded x{was}). "
                "An undefined name is a NameError on the branch that reaches it — "
                "bind it or import it; the baseline may only go down."
            )
        elif now < was:
            if _is_absent(rel):
                absent.append(f"{rel}: `{name}` x{was} — file not in this tree.")
            else:
                problems.append(
                    f"{rel}: `{name}` is down to x{now} from x{was} — lower the "
                    f"baseline (`--write`). A stale row pre-exempts the name if it "
                    "comes back."
                )
    for rel in sorted(unparseable - known_unparseable):
        problems.append(
            f"{rel}: does not parse. Every AST gate in this repo skips it "
            "(`except SyntaxError: continue`), so it is silently exempt from all "
            "of them — this is not merely a broken file."
        )
    for rel in sorted(known_unparseable - unparseable):
        if _is_absent(rel):
            absent.append(f"{rel}: recorded unparseable — file not in this tree.")
        else:
            problems.append(f"{rel}: now parses — remove it from {UNPARSEABLE_FILENAME}.")
    return problems, absent


def render(counts: Counter[tuple[str, str]]) -> str:
    header = (
        "# F821 (undefined name) ratchet baseline — GENERATED.\n"
        "# Regenerate ONLY to lower it: python scripts/ci/check_f821_ratchet.py --write\n"
        "# Format: count<TAB>path<TAB>name. Counted, not just identified, because\n"
        "# identity alone lets a second undefined name land beside a recorded one.\n"
    )
    rows = "".join(f"{n}\t{rel}\t{name}\n" for (rel, name), n in sorted(counts.items()))
    return header + rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--write", action="store_true", help="Re-record the baseline.")
    parser.add_argument(
        "--absent-files-ok",
        action="store_true",
        help="Do not fail on a baseline row whose FILE is missing from --root. For "
        "a tree that deliberately ships a subset (the public export). Rows whose "
        "file is present are still checked in full, and every absence is printed.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent / "baselines"
    baseline_path, unparseable_path = here / BASELINE_FILENAME, here / UNPARSEABLE_FILENAME

    found, unparseable = split_diagnostics(_run_ruff(args.root), args.root)
    if args.write:
        here.mkdir(exist_ok=True)
        baseline_path.write_text(render(found), encoding="utf-8")
        unparseable_path.write_text(
            "# Files that do not parse, so their F821 status is UNKNOWN and every\n"
            "# AST gate in the repo silently skips them. Each one is a bug (#1404).\n"
            + "".join(f"{r}\n" for r in sorted(unparseable)),
            encoding="utf-8",
        )
        print(
            f"Recorded {sum(found.values())} F821 in {len(found)} (file, name) pairs; "
            f"{len(unparseable)} unparseable file(s)."
        )
        return 0

    problems, absent = compare(
        found,
        read_baseline(baseline_path),
        unparseable,
        read_lines(unparseable_path),
        root=args.root,
        absent_files_ok=args.absent_files_ok,
    )
    if absent:
        print(f"{len(absent)} baseline row(s) name a file absent from {args.root}:")
        for a in absent:
            print(f"  {a}")
    if problems:
        print("F821 ratchet violated (CLAUDE.md non-negotiable #19):")
        for p in problems:
            print(f"  {p}")
        print(f"\n{len(problems)} problem(s).")
        return 1
    print(
        f"F821 ratchet: OK ({sum(found.values())} recorded undefined name(s) in "
        f"{len(found)} (file, name) pair(s); {len(unparseable)} unparseable file(s))."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
