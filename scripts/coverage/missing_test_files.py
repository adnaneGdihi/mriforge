#!/usr/bin/env python
"""List src/mriforge modules that are not referenced by any test file.

For every ``.py`` file under ``src/mriforge/`` that:
  - is not ``__init__.py``
  - has at least 40 non-blank, non-comment lines

…search whether *any* ``tests/**/test_*.py`` file contains the module's
stem as a substring of its content (e.g. ``fft_ops`` appears somewhere
inside the test file's text).

Output (CSV to stdout)::

    module_path,referenced,layer

A summary count of unreferenced modules is printed to stderr at the end.

Usage::

    python scripts/coverage/missing_test_files.py
    python scripts/coverage/missing_test_files.py --min-loc 20
    python scripts/coverage/missing_test_files.py --unreferenced-only

Exits 0 always — this is a **reporting tool**, not a CI gate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "mriforge"
TESTS_ROOT = REPO_ROOT / "tests"

_MIN_LOC_DEFAULT = 40


def _count_substantive_lines(path: Path) -> int:
    """Return non-blank, non-comment line count for *path*."""
    count = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    except OSError:
        pass
    return count


def _build_test_corpus() -> dict[str, str]:
    """Return {test_file_path_str: file_contents} for all tests/**/test_*.py."""
    corpus: dict[str, str] = {}
    for tp in TESTS_ROOT.rglob("test_*.py"):
        if "__pycache__" in tp.parts:
            continue
        try:
            corpus[str(tp)] = tp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return corpus


def _layer_of(module_path: Path) -> str:
    """Return the two-component layer label (relative to src/mriforge/).

    Examples::

        src/mriforge/infrastructure/physics/fft_ops.py -> infrastructure/physics
        src/mriforge/models/generators/unet.py          -> models/generators
        src/mriforge/main.py                            -> (root)
        src/mriforge/cli/app.py                         -> cli
    """
    rel = module_path.relative_to(SRC_ROOT)
    parts = rel.parts[:-1]  # drop filename
    if len(parts) == 0:
        return "(root)"
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}/{parts[1]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List src/mriforge modules not referenced by any test file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=_MIN_LOC_DEFAULT,
        metavar="N",
        help=f"minimum substantive LOC to include a module (default: {_MIN_LOC_DEFAULT})",
    )
    parser.add_argument(
        "--unreferenced-only",
        action="store_true",
        help="emit only rows where referenced=False",
    )
    args = parser.parse_args(argv)

    if not SRC_ROOT.is_dir():
        print(f"error: src root not found at {SRC_ROOT}", file=sys.stderr)
        return 2

    # Build test corpus once (avoid re-reading for every module).
    corpus = _build_test_corpus()

    # Enumerate qualifying source modules.
    modules: list[Path] = []
    for py in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        if py.name == "__init__.py":
            continue
        if _count_substantive_lines(py) < args.min_loc:
            continue
        modules.append(py)

    # For each module, check if its stem appears in any test file content.
    rows: list[tuple[str, bool, str]] = []
    for module in modules:
        stem = module.stem  # e.g. 'fft_ops'
        referenced = any(stem in content for content in corpus.values())
        layer = _layer_of(module)
        rel = str(module.relative_to(REPO_ROOT))
        rows.append((rel, referenced, layer))

    # Write CSV to stdout.
    writer = csv.writer(sys.stdout)
    writer.writerow(["module_path", "referenced", "layer"])
    for rel, referenced, layer in rows:
        if args.unreferenced_only and referenced:
            continue
        writer.writerow([rel, str(referenced), layer])

    # Summary to stderr.
    total = len(rows)
    unreferenced = sum(1 for _, ref, _ in rows if not ref)
    if total == 0:
        # No module cleared the LOC floor. Guard the percentage: this used to be
        # an unconditional `unreferenced / total`, so the script CRASHED with
        # ZeroDivisionError instead of reporting an empty scan (e.g. --min-loc
        # above every module's size, or a src tree of stubs).
        print(
            f"\nSummary: 0 modules scanned (min_loc>={args.min_loc}) — "
            f"nothing cleared the LOC floor.",
            file=sys.stderr,
        )
        return 0

    print(
        f"\nSummary: {total} modules scanned (min_loc>={args.min_loc}), "
        f"{unreferenced} unreferenced ({100.0 * unreferenced / total:.1f}% of total)",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
