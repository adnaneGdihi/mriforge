#!/usr/bin/env python
"""Print a per-layer coverage breakdown from a coverage.xml file.

Parses the XML produced by ``coverage.py --format xml`` (Cobertura schema).
Groups every ``<class filename=...>`` entry by the **first two path components
under the package root** (e.g. ``infrastructure/physics/x.py`` → layer
``infrastructure/physics``; ``main.py`` → ``main.py``).

Sums valid/covered line counts per group, then prints a table sorted by
missing lines descending::

    cover% |   miss |  valid | n_files | layer
    -------+--------+--------+---------+-------------------------------
     13.8% |   1820 |   2113 |     113 | infrastructure/physics
      8.3% |   1445 |   1577 |     192 | models/generators
     ...

Usage::

    python scripts/coverage/print_per_layer.py tests_experiments/coverage.xml
    python scripts/coverage/print_per_layer.py coverage.xml --min-valid 50

Exits 0 always — this is a **reporting tool**, not a CI gate.

``layer_rows`` and ``format_table`` are the reusable half: the SLURM array's
aggregated report renders the same table through them
(``scripts/ci/coverage_summary.py``) rather than regrouping the XML itself, so
there is exactly one owner of what "a layer" means.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

#: Package root component. Coverage writes ``<class filename=...>`` RELATIVE to
#: the ``<source>`` root, and this repo's ``[tool.coverage.run] source`` is
#: ``src/spectramr`` — so filenames normally arrive already package-relative
#: (``infrastructure/physics/x.py``) with no prefix to strip. A run configured
#: with ``source = ["src"]`` instead yields ``spectramr/infrastructure/...``.
#: Both shapes must map to the same layer; see ``_layer_of``.
_PACKAGE = "spectramr"


class LayerRow(NamedTuple):
    """One row of the per-layer table."""

    layer: str
    pct: float
    miss: int
    valid: int
    n_files: int
    covered: int


def _layer_of(filename: str) -> str:
    """Map a class filename to its two-component layer label.

    Rules:
    - Strip a leading ``spectramr/`` component **only if it is actually there**.
    - Keep the next two components if they are both directory parts.
    - If only one component remains (top-level file), use that filename.

    Examples::

        infrastructure/physics/fft_ops.py          -> infrastructure/physics
        spectramr/infrastructure/physics/fft_ops.py  -> infrastructure/physics
        models/generators/unet.py                  -> models/generators
        main.py                                    -> main.py
        cli/app.py                                 -> cli/app.py

    The conditional strip is load-bearing. This function used to drop
    ``parts[0]`` unconditionally, but with ``source = ["src/spectramr"]`` the XML
    carries no package prefix, so it was eating a **real** layer: every
    ``infrastructure/training/strategies/*`` file was labelled
    ``training/strategies``, ``cli/app.py`` collapsed to ``app.py``, and
    ``core/metrics/*`` and ``models/metrics/*`` merged into one fictitious
    ``metrics/*`` row whose percentage described neither.
    """
    parts = Path(filename).parts  # ('infrastructure', 'physics', 'fft_ops.py')
    if len(parts) == 0:
        return "(unknown)"
    sub = parts[1:] if parts[0] == _PACKAGE else parts
    if len(sub) == 0:
        # The filename was exactly 'spectramr' — nothing under it to name.
        return parts[0]
    if len(sub) == 1:
        # top-level file like 'main.py'
        return sub[0]
    # First two remaining components (may be dir/dir, dir/file, etc.)
    return f"{sub[0]}/{sub[1]}"


def layer_rows(xml_path: Path, min_valid: int = 0) -> list[LayerRow]:
    """Parse ``xml_path`` into per-layer rows, sorted by missing lines desc.

    Raises ``FileNotFoundError`` if the XML is absent and ``ValueError`` if it
    parses but holds no measured lines — an empty report must never be
    presentable as 0 % coverage, which reads as a real measurement.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"coverage.xml not found at {xml_path}")

    root = ET.parse(xml_path).getroot()

    layer_valid: dict[str, int] = defaultdict(int)
    layer_covered: dict[str, int] = defaultdict(int)
    layer_files: dict[str, set[str]] = defaultdict(set)

    for cls in root.iter("class"):
        filename = cls.get("filename", "")
        layer = _layer_of(filename)
        layer_files[layer].add(filename)
        for line in cls.iter("line"):
            layer_valid[layer] += 1
            if line.get("hits", "0") != "0":
                layer_covered[layer] += 1

    if not layer_valid:
        raise ValueError(f"no coverage data found in {xml_path}")

    rows = [
        LayerRow(
            layer=layer,
            pct=100.0 * layer_covered.get(layer, 0) / valid if valid else 0.0,
            miss=valid - layer_covered.get(layer, 0),
            valid=valid,
            n_files=len(layer_files[layer]),
            covered=layer_covered.get(layer, 0),
        )
        for layer, valid in layer_valid.items()
        if valid >= min_valid
    ]
    rows.sort(key=lambda r: -r.miss)
    return rows


def totals(rows: list[LayerRow]) -> LayerRow:
    """Aggregate ``rows`` into a single TOTAL row.

    Computed from the rows actually shown, so a ``--min-valid`` filter yields a
    TOTAL consistent with the table above it rather than a repo-wide figure the
    visible rows do not sum to.
    """
    valid = sum(r.valid for r in rows)
    covered = sum(r.covered for r in rows)
    return LayerRow(
        layer="TOTAL",
        pct=100.0 * covered / valid if valid else 0.0,
        miss=valid - covered,
        valid=valid,
        n_files=sum(r.n_files for r in rows),
        covered=covered,
    )


def format_table(rows: list[LayerRow], include_total: bool = True) -> str:
    """Render ``rows`` as the fixed-width table, without printing it."""
    layer_w = max([len(r.layer) for r in rows] + [len("layer"), len("TOTAL")])
    header = f"{'cover%':>7} | {'miss':>7} | {'valid':>7} | {'n_files':>7} | {'layer':<{layer_w}}"
    sep = "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * layer_w

    def _fmt(row: LayerRow) -> str:
        return (
            f"{row.pct:>6.1f}% | {row.miss:>7d} | {row.valid:>7d} | "
            f"{row.n_files:>7d} | {row.layer:<{layer_w}}"
        )

    out = [header, sep, *(_fmt(r) for r in rows)]
    if include_total:
        out += [sep, _fmt(totals(rows))]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print per-layer coverage summary from coverage.xml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="tests_experiments/coverage.xml",
        help="path to coverage.xml (default: tests_experiments/coverage.xml)",
    )
    parser.add_argument(
        "--min-valid",
        type=int,
        default=0,
        metavar="N",
        help="hide layers with fewer than N valid lines (default: 0)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="omit the TOTAL row",
    )
    args = parser.parse_args(argv)

    try:
        rows = layer_rows(Path(args.xml_path), min_valid=args.min_valid)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("no layers matched the filter criteria.", file=sys.stderr)
        return 1

    print(format_table(rows, include_total=not args.no_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
