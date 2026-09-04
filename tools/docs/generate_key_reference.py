#!/usr/bin/env python3
"""Generate ``docs/config_key_reference.rst`` from the rename SSOT.

Every retired config key, where it went, and why -- written from
``spectramr.config.schemas.renames.RENAMES`` rather than maintained by hand, so
the page cannot drift from the table the schema actually enforces. A hand-kept
list of renames is the same defect class the table exists to remove, one level
up: it would send an author to a spelling that no longer resolves.

``tests/unit/config/test_key_reference_is_current.py`` regenerates and compares,
so a rename that lands without running this fails CI rather than silently
shipping a stale page.

Usage::

    python tools/docs/generate_key_reference.py            # write the page
    python tools/docs/generate_key_reference.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from spectramr.config.schemas.renames import RENAMES, ROOT  # noqa: E402

OUTPUT = _REPO_ROOT / "docs" / "config_key_reference.rst"

_HEADER = """.. _config_key_reference:

=====================================
Retired Configuration Keys
=====================================

.. warning::

   **This page is generated.** Edit
   ``spectramr/config/schemas/renames.py`` and run
   the page generator in the maintainers' tree; do not edit the tables below.

Every key that has moved, where it went, and why. The rename table is the single
source of truth for four things at once -- the schema shim that accepts or
rejects the old spelling, the fixer that rewrites YAML, the corpus gate that
counts what is left, and the ``--override`` path translation -- so a key listed
here behaves identically in all four.

Two postures
============

**fold** -- staged. The old spelling still LOADS: a ``mode="before"`` validator
moves the value to its canonical path before validation. It is gone from Python,
so there is one read path and, temporarily, two accepted spellings in YAML. Fold
records are what let a rename land before the corpus migration.

**raise** -- retired. The old spelling is an error naming its replacement. A
``raise`` record only lands together with the migration that drives its corpus
usage to zero.

Fix any of these automatically::

    python scripts/ci/migrate_config_keys.py <your-arm>.yaml --apply


What the fixer cannot reach
===========================

The fixer edits *lines*, not a parsed document -- a ruamel round-trip reflows the
whole file and buries a rename under a diff nobody reviews. The cost is that it
cannot enter a flow mapping::

    validation: {metrics: [psnr], val_batch_size: 4}

A key in that shape used to be reported as **absent**, which is the same answer
the tool gives for a file that never declared it. That silence mattered because
the promotion rule reads a count: drive a fold record to zero, flip its posture to
``raise``, delete the shim -- and every arm hiding the key in a flow mapping then
fails at load.

Both migrators now run a parser-based **detector** alongside the line-scanning
**rewriter**, and refuse when they disagree:

``UNSUPPORTED``
    Declared, and unreachable. Reported **separately from the STAGED countdown**
    and non-zero exit. A record may not be promoted while any remain.

Reflow the offending file to block style, then re-run. For the live count, run
either migrator over the corpus -- this page is generated from ``RENAMES`` alone
and does not scan ``experiments/``, so a number written here would only drift.
"""


def _rows(posture: str) -> list[tuple[str, str, str, str]]:
    out = []
    for rec in RENAMES.values():
        if rec.posture != posture:
            continue
        legacy = rec.legacy if rec.block != ROOT else f"{rec.legacy} *(root)*"
        note = rec.reason.strip()
        if rec.value_transform == "negate":
            note = "**SENSE INVERTED -- negate the value too.** " + note
        out.append((legacy, rec.canonical, rec.since, note))
    return sorted(out)


def _table(rows: list[tuple[str, str, str, str]]) -> str:
    if not rows:
        return "*(none)*\n"
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 26 26 10 38",
        "",
        "   * - Retired key",
        "     - Replacement",
        "     - Since",
        "     - Why",
    ]
    for legacy, canonical, since, note in rows:
        lines += [
            f"   * - ``{legacy}``",
            f"     - ``{canonical}``",
            f"     - {since}",
            f"     - {note}",
        ]
    return "\n".join(lines) + "\n"


def render() -> str:
    folds, raises = _rows("fold"), _rows("raise")
    return (
        _HEADER
        + f"\nRetired outright ({len(raises)})\n"
        + "=" * (len(f"Retired outright ({len(raises)})"))
        + "\n\n"
        + "These raise on load. Write the replacement.\n\n"
        + _table(raises)
        + f"\nStaged -- the old spelling still loads ({len(folds)})\n"
        + "=" * (len(f"Staged -- the old spelling still loads ({len(folds)})"))
        + "\n\n"
        + "Accepted for now and folded into place. Migrate at your convenience;\n"
        + "``scripts/ci/check_no_legacy_config_keys.py`` prints how many remain.\n\n"
        + _table(folds)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the page is stale")
    args = ap.parse_args(argv)
    new = render()
    if args.check:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != new:
            print(f"{OUTPUT} is stale -- run python {Path(__file__).name}")
            return 1
        print(f"{OUTPUT} is current ({len(RENAMES)} records)")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(new)
    print(f"wrote {OUTPUT} ({len(RENAMES)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
