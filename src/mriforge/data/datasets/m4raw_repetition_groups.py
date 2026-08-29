"""M4Raw repetition-group discovery (data SSOT).

Grouping ``.h5`` acquisition files by subject+contrast (stripping the trailing
repetition number from the stem) is a file→dataset concern, so per the
data-loading SSOT (CLAUDE.md pitfall #11 / #13) it lives under
``src/mriforge/data/`` — *not* in ``scripts/sim2rank``. The sim2rank engine and the
``mriforge meta-evaluate`` CLI both import :func:`discover_repetition_groups` from
here, so neither has to reach across the layer boundary (``src/`` importing from
``scripts/`` is a merge-blocking violation).

The grouping mirrors ``M4RawRepetitionDataset._build_groups``: files whose stems
share everything but the last two characters (the repetition index, e.g. ``01``,
``02``) form one multi-repetition group.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["discover_repetition_groups"]


def discover_repetition_groups(
    data_dir: Path,
    max_subjects: int = 5,
    max_contrasts: int = 3,
) -> list[list[Path]]:
    """Discover repetition groups from an m4raw directory.

    Groups files by stripping the last 2 characters of the stem (the repetition
    number, e.g. ``01``, ``02``). This mirrors the proven logic in
    ``M4RawRepetitionDataset._build_groups``.

    Args:
        data_dir: directory holding the ``.h5`` acquisitions (searched
            non-recursively first, then recursively).
        max_subjects: cap on subjects (combined with ``max_contrasts`` to bound
            the number of returned groups).
        max_contrasts: cap on contrasts per subject.

    Returns:
        A list of repetition groups, each a sorted list of file paths. Prefers
        multi-repetition groups; falls back to singletons when none exist.

    Raises:
        FileNotFoundError: when no ``.h5`` files are found under ``data_dir``.
    """
    h5_files = sorted(data_dir.glob("*.h5"))
    if not h5_files:
        h5_files = sorted(data_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {data_dir}")

    groups: dict[str, list[Path]] = {}
    for f in h5_files:
        stem = f.name.split(".")[0]  # strip ALL extensions
        # Strip the last 2 chars (rep number); too-short stems stay singletons.
        base = stem if len(stem) < 3 else stem[:-2]
        groups.setdefault(base, []).append(f)

    # Prefer multi-rep groups; fall back to singletons
    multi_rep_groups = [sorted(p) for p in groups.values() if len(p) >= 2]
    if not multi_rep_groups:
        logger.warning("No multi-repetition groups found. Using single files.")
        multi_rep_groups = [[f] for f in h5_files]

    logger.info(
        "Discovered %d repetition groups (%d multi-rep) from %d H5 files",
        len(groups),
        len(multi_rep_groups),
        len(h5_files),
    )
    return multi_rep_groups[: max_subjects * max_contrasts]
