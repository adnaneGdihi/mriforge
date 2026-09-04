"""fastMRI-brain contrast-aware group discovery (data SSOT).

Unlike M4Raw (several phase-incoherent repetitions per anatomy), fastMRI brain
stores **one fully-sampled acquisition per file**, tagged with a contrast
(``AXT1`` / ``AXT2`` / ``AXT2FLAIR`` / ``AXFLAIR`` / ``AXT1POST`` / ``AXT1PRE``).

sim2rank's native path assigns ``(subject, contrast)`` by *flat filename-order
index* (``group_idx = s * n_contrasts + c``). fastMRI names sort **contrast
-major** (``file_brain_AXT1_*`` < ``AXT2_*`` < ``AXT2FLAIR_*``), so passing brain
data straight through ``discover_repetition_groups`` chops three *same-contrast*
files onto one subject's "three contrasts" and the per-contrast LOCO concordance
grades mislabelled buckets (issue #308).

This module builds a **balanced, interleaved** grid instead: each ``.h5`` is its
own single-rep reference, and the returned order is subject-major / contrast
-minor across the chosen contrasts, so ``c_idx`` maps to a *real* contrast. The
grid dimensions are returned explicitly because sim2rank's ``n_subjects`` /
``n_contrasts`` heuristic (derived from ``len(rep_groups)``) assumes M4Raw
ordering and cannot recover a C-wide grid on its own.

Grouping / labelling is a file->dataset concern, so per the data-loading SSOT
(CLAUDE.md pitfall #11 / #13) it lives here, not in ``scripts/sim2rank``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py

from spectramr.infrastructure.physics.m4raw_pseudo_gt import extract_contrast_label

logger = logging.getLogger(__name__)

__all__ = ["BrainContrastGrid", "brain_contrast_of", "discover_brain_contrast_groups"]


@dataclass(frozen=True)
class BrainContrastGrid:
    """A rectangular ``n_subjects x n_contrasts`` grid of single-acquisition refs.

    ``groups`` is subject-major / contrast-minor so that ``group_idx =
    s * n_contrasts + c`` selects volume ``s`` of contrast ``contrasts[c]`` — the
    ordering sim2rank's ``(subject, contrast)`` loop expects. Each entry is a
    one-element list (one fully-sampled acquisition = one clean reference).
    """

    groups: list[list[Path]]
    n_subjects: int
    n_contrasts: int
    contrasts: list[str]


def brain_contrast_of(path: Path) -> str:
    """Contrast for a fastMRI-brain file.

    Prefers the authoritative ``attrs['acquisition']`` written by fastMRI; falls
    back to the ``AX*`` filename token via :func:`extract_contrast_label`. Raises
    when neither yields a recognisable contrast rather than mislabelling the
    volume with its full filename (no silent fallback, pitfall #9).
    """
    acquisition: str | None = None
    try:
        with h5py.File(str(path), "r") as f:
            raw = f.attrs.get("acquisition")
        if raw is not None:
            acquisition = raw.decode() if isinstance(raw, bytes) else str(raw)
    except (OSError, KeyError) as exc:  # unreadable / no attrs -> try the filename
        logger.debug("No readable acquisition attr in %s (%s); using filename.", path, exc)

    if acquisition:
        return acquisition.strip().upper()

    label = extract_contrast_label(path)
    if label == path.stem:
        raise ValueError(
            f"Cannot determine a contrast for {path.name}: no 'acquisition' attr "
            f"and no recognisable AX* / M4Raw token in the filename."
        )
    return label


def discover_brain_contrast_groups(
    data_dir: Path,
    max_subjects: int = 5,
    max_contrasts: int = 3,
    contrasts: list[str] | None = None,
) -> BrainContrastGrid:
    """Discover a balanced contrast grid from a directory of fastMRI-brain ``.h5``.

    Args:
        data_dir: directory of single-acquisition ``.h5`` files (searched
            non-recursively first, then recursively).
        max_subjects: cap on volumes *per contrast* (the grid's ``n_subjects``).
        max_contrasts: cap on the number of contrasts (the grid's ``n_contrasts``).
        contrasts: optional explicit allow-list (e.g. ``["AXT1", "AXT2",
            "AXT2FLAIR"]``). When given, every requested contrast must be present
            in the data or a ``ValueError`` is raised (fail-loud). When omitted,
            the ``max_contrasts`` most-populated contrasts are chosen.

    Returns:
        A :class:`BrainContrastGrid`. The grid is balanced (``n_subjects`` =
        smallest chosen bucket, capped by ``max_subjects``); volumes beyond the
        rectangle are dropped and the drop count is logged (no silent cap).

    Raises:
        FileNotFoundError: no ``.h5`` under ``data_dir``.
        ValueError: a requested contrast is absent, or a chosen bucket is empty.
    """
    h5_files = sorted(data_dir.glob("*.h5")) or sorted(data_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {data_dir}")

    buckets: dict[str, list[Path]] = {}
    for f in h5_files:
        buckets.setdefault(brain_contrast_of(f), []).append(f)

    if contrasts:
        requested = [c.strip().upper() for c in contrasts]
        missing = [c for c in requested if c not in buckets]
        if missing:
            raise ValueError(
                f"Requested contrast(s) {missing} not found in {data_dir}. "
                f"Present: {sorted(buckets)}"
            )
        chosen = requested[:max_contrasts]
    else:
        # Most-populated first, name as a deterministic tie-break.
        chosen = sorted(buckets, key=lambda c: (-len(buckets[c]), c))[:max_contrasts]

    if not chosen:
        raise ValueError(f"No usable contrasts discovered in {data_dir}")

    per_contrast = min(min(len(buckets[c]) for c in chosen), max_subjects)
    if per_contrast < 1:
        raise ValueError(f"Chosen contrasts {chosen} have no volumes to grade.")

    total = sum(len(buckets[c]) for c in chosen)
    used = per_contrast * len(chosen)
    if used < total:
        logger.info(
            "fastMRI-brain grid balanced to %d/%d volumes (%d per contrast x %d "
            "contrasts); dropped %d to keep a rectangular subject x contrast grid.",
            used,
            total,
            per_contrast,
            len(chosen),
            total - used,
        )

    groups: list[list[Path]] = []
    for s in range(per_contrast):
        for c in chosen:
            groups.append([buckets[c][s]])

    logger.info(
        "Discovered fastMRI-brain grid: %d subjects x %d contrasts %s (%d h5 files).",
        per_contrast,
        len(chosen),
        chosen,
        len(h5_files),
    )
    return BrainContrastGrid(
        groups=groups,
        n_subjects=per_contrast,
        n_contrasts=len(chosen),
        contrasts=chosen,
    )
