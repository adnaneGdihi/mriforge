"""Read a corpus YAML key under whichever spelling the arm actually uses.

The key drain rewrites retired spellings to their canonical paths one cohort
at a time, so for as long as it runs **both spellings are in the corpus** --
`mrixfields2026` says `validation.loader.num_batches`, the other 58 cohorts
still say `validation.num_validation_batches`, and a repo-wide guard meets
both on the same sweep.

A guard that hardcodes one spelling is wrong in one direction or the other at
every point during the drain, and wrong *silently*: the lookup returns `None`,
which reads as "the arm does not declare this" rather than "I looked in the
wrong place". That is what happened when `mrixfields2026` was drained -- a
regression guard protecting against a cluster OOM that had killed 70 arms
started reporting the cap as unset on all 77 of them, and its sibling sweep
collapsed from 29 arms to 2.

This resolves through :data:`RENAMES`, the same table the schema and the
migrator use, so a guard built on it follows the next rename automatically
instead of needing a second edit that someone has to remember.

Scope: raw ``yaml.safe_load`` documents. Code holding a resolved
``TrainingSettings`` already has one spelling and should read the canonical
attribute path directly -- see the nested-key rules in ``CLAUDE.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mriforge.config.schemas.renames import RENAMES

_MISSING = object()


def legacy_spellings(canonical: str) -> list[str]:
    """Every still-folding legacy path that resolves to ``canonical``.

    Only ``fold`` records are returned. A ``raise`` record is retired: an arm
    declaring it does not load at all, so a guard must NOT accept it -- doing
    so would report a config as healthy that cannot start.
    """
    return sorted(
        record.legacy
        for record in RENAMES.values()
        if record.canonical == canonical and record.posture == "fold"
    )


def _dig(doc: Any, path: str) -> Any:
    node = doc
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def read_key(doc: Any, canonical: str, *, default: Any = None) -> Any:
    """Value at ``canonical``, falling back to any legacy spelling that folds.

    Canonical wins when both are present, matching the loader: the fold only
    fills a destination the arm left empty.
    """
    found = _dig(doc, canonical)
    if found is not _MISSING:
        return found
    for legacy in legacy_spellings(canonical):
        found = _dig(doc, legacy)
        if found is not _MISSING:
            return found
    return default


def spelling_used(doc: Any, canonical: str) -> str | None:
    """Which path the arm actually declares, or ``None`` if it declares neither.

    For anti-vacuity assertions. A guard that accepts both spellings needs to
    prove both branches are live, or a broken canonical lookup hides behind the
    legacy fallback until the drain finishes and everything fails at once.
    """
    if _dig(doc, canonical) is not _MISSING:
        return canonical
    for legacy in legacy_spellings(canonical):
        if _dig(doc, legacy) is not _MISSING:
            return legacy
    return None
