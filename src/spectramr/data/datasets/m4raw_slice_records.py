"""Slice-level records for :class:`~spectramr.data.datasets.m4raw_dataset.M4RawRepetitionDataset`.

The default M4Raw index holds one record per (patient, contrast) group, and
serving one ``[H, W, 1]`` patch from it loads every repetition of the group in
full: ``slices / samples_per_volume`` reads per served slice, times the
repetition count, every epoch (#1757). ``data.slice_level_records`` expands each
group into one record per slice, so a served sample costs one slice read per
repetition and the dataset's ``__len__`` becomes the corpus slice count.

This module is the pure half: it turns group records into slice records from
header-only shape reads and never opens voxel data. The loader half
(``_load_kspace(path, slice_index)``) lives in the dataset module.

A slice record is the group record plus:

``group_index``
    Position of the parent group in the original index, so group-level counts
    (``provenance_counts``) and the retry step in ``__getitem__`` can tell
    records of one group apart from records of different groups.
``slice_index`` / ``num_slices``
    The slice this record serves and the group's slice count.
``shape``
    The parent's raw k-space shape with the leading (slice) axis set to 1. The
    queue builder's patch-compatibility fast path reads ``shape[0]`` against the
    patch depth, so a depth-3 patch drops every slice record instead of
    reaching the sampler with a depth-1 subject.

Queue decision (recorded here because the answer must not double-sample): the
TorchIO queue is **not** bypassed. With ``samples_per_volume: 1`` and a depth-1
patch, ``tio.Queue`` draws exactly one patch per record and
``iterations_per_epoch == len(records)``, so every slice is served once per
epoch. The ``full`` sampler bypass serves each record once as well. Any other
``samples_per_volume`` draws that many patches of the same slice per epoch --
on M4Raw's native 256 x 256 matrix a full-slice patch makes them identical
copies -- and the audit check ``slice_level_records_queue_shape`` refuses it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Header-only shape reader: ``(S, [C,] H, W)`` or ``None`` when unreadable.
ShapeReader = Callable[[str | Path], tuple[int, ...] | None]

#: The file lists a group record may carry, in the order
#: ``M4RawRepetitionDataset._attach_shape_metadata`` consults them. The first
#: non-empty side is the one whose header stamped ``record["shape"]``.
_SIDES: tuple[str, ...] = ("source", "paths", "target")


def _label(record: dict[str, Any]) -> str:
    """Human-readable name for a group in an error message."""
    patient = record.get("patient_id")
    contrast = record.get("contrast") or record.get("target_contrast")
    if patient is not None and contrast is not None:
        return f"{patient}/{contrast}"
    for side in _SIDES:
        files = record.get(side) or ()
        if files:
            return Path(str(files[0])).name
    return "<empty record>"


def slice_count_of(record: dict[str, Any], read_shape: ShapeReader) -> tuple[int, tuple[int, ...]]:
    """The slice count a group promises on every side, and its stamped shape.

    The first non-empty side reuses ``record["shape"]`` (already read once by
    ``_attach_shape_metadata``); every further side is read from its first
    file's header. Sides must agree: a federated pair whose source and target
    volumes differ in slice count has no per-slice pairing.

    Returns:
        ``(num_slices, shape)`` where ``shape`` is the leading side's raw
        k-space shape, slice axis first.

    Raises:
        ValueError: a side's header could not be read, the shape has no slice
            axis, or the sides disagree.
    """
    counts: dict[str, int] = {}
    leading_shape: tuple[int, ...] | None = None
    for side in _SIDES:
        files = record.get(side) or ()
        if not files:
            continue
        shape = record.get("shape") if leading_shape is None else read_shape(files[0])
        if shape is None:
            raise ValueError(
                f"[M4Raw] slice_level_records: the k-space header of {Path(str(files[0])).name} "
                f"(group {_label(record)}, side {side!r}) could not be read, so its slice "
                "count is unknown. The slice-level index needs every group's slice count; "
                "fix or drop the file rather than guessing a count."
            )
        if len(shape) < 3:
            raise ValueError(
                f"[M4Raw] slice_level_records: {Path(str(files[0])).name} (group "
                f"{_label(record)}) stores k-space as {tuple(shape)}, which has no slice "
                "axis to index; the slice-level route needs (S, [C,] H, W) volumes."
            )
        if leading_shape is None:
            leading_shape = tuple(shape)
        counts[side] = int(shape[0])
    if leading_shape is None:
        raise ValueError(f"[M4Raw] slice_level_records: record {_label(record)} names no files.")
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{side}={n}" for side, n in counts.items())
        raise ValueError(
            f"[M4Raw] slice_level_records: group {_label(record)} has different slice "
            f"counts per side ({detail}); a per-slice pairing needs them equal."
        )
    return next(iter(counts.values())), leading_shape


def expand_to_slice_records(
    groups: list[dict[str, Any]], read_shape: ShapeReader
) -> list[dict[str, Any]]:
    """One record per (group, slice), group-major, slice ascending.

    ``len(result) == sum(slice count over groups)``. Every problem is
    collected and raised once so a run with several unreadable headers reports
    all of them rather than the first.
    """
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for group_index, group in enumerate(groups):
        try:
            num_slices, shape = slice_count_of(group, read_shape)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        slice_shape = (1, *shape[1:])
        for slice_index in range(num_slices):
            records.append(
                {
                    **group,
                    "group_index": group_index,
                    "slice_index": slice_index,
                    "num_slices": num_slices,
                    "shape": slice_shape,
                }
            )
    if problems:
        raise ValueError(
            f"[M4Raw] slice_level_records could not index {len(problems)} of "
            f"{len(groups)} group(s):\n  " + "\n  ".join(problems)
        )
    return records


def retry_step(record: dict[str, Any]) -> int:
    """How far ``__getitem__`` advances to leave this record's group on a skip.

    A slice record's neighbours are the other slices of the same files, so a
    step of one would retry the same unreadable group and report a systemic
    failure after five attempts. A group record steps by one, as before.
    """
    return max(1, int(record.get("num_slices") or 1))
