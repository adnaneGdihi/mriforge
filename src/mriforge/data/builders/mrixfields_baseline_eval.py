"""Paired full-volume eval loader for the MRIxFields2026 baseline harness.

This module is the **data-layer SSOT** for reading NIfTI volumes from the
MRIxFields2026 challenge dataset and yielding paired (source, target)
full-volume arrays for a given (source_field, target_field, contrast)
specialization.

All nibabel I/O lives here; nothing above the data layer may call
``nib.load`` on real NIfTI files (per the repo's data-SSOT rule, pitfall #11).

Volumes are expected to be in **[0, 1]** range (as stored by the challenge).
The loader warns (but does not rescale) if a volume falls outside [-0.5, 1.5].
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairedVolume:
    """A single paired (source, target) full volume for one subject."""

    subject_id: str
    source: np.ndarray  # [H, W, D] float32, values in [0, 1]
    target: np.ndarray  # [H, W, D] float32, values in [0, 1]
    affine: np.ndarray  # 4x4 float64 -- canonical-space affine of the source volume
    voxel_size: tuple[float, float, float]  # mm, from source header
    source_path: Path
    target_path: Path


def load_manifest_records(path) -> list[dict]:
    """Load the ``"records"`` list from a MRIxFields2026 manifest JSON.

    Parameters
    ----------
    path:
        Path to the manifest JSON file.  The file must have the structure
        ``{"records": [...]}`` where each record carries at minimum
        ``relative_path``, ``field_strength``, ``contrast``, and
        ``subject_id``.

    Returns
    -------
    list[dict]
        The raw record list.
    """
    with open(path) as f:
        return json.load(f)["records"]


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI volume in RAS-canonical orientation.

    Returns ``(data, affine, voxel_size)`` where *data* is ``float32`` and
    *voxel_size* is a 3-tuple of mm values from the header.  A ``WARNING``
    is emitted if the voxel intensities fall outside ``[-0.5, 1.5]``.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    zooms = img.header.get_zooms()[:3]
    vox = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    if data.max() > 1.5 or data.min() < -0.5:
        logger.warning(
            "%s outside expected [0,1] (min=%.3g max=%.3g)",
            path.name,
            float(data.min()),
            float(data.max()),
        )
    return data, np.asarray(img.affine), vox


def _emit_pair(data_root: Path, src_r: dict, tgt_r: dict, subject_id: str) -> PairedVolume:
    """Load a (source, target) record pair into a :class:`PairedVolume`."""
    s, aff, vox = _load_canonical(data_root / src_r["relative_path"])
    t, _, _ = _load_canonical(data_root / tgt_r["relative_path"])
    return PairedVolume(
        subject_id=subject_id,
        source=s,
        target=t,
        affine=aff,
        voxel_size=vox,
        source_path=data_root / src_r["relative_path"],
        target_path=data_root / tgt_r["relative_path"],
    )


def iter_paired_volumes(
    records: list[dict],
    data_root,
    *,
    source_field: float,
    target_field: float,
    contrast: str,
    split: str = "Validating_prospective",
    pairing: str = "ordinal",
) -> Iterator[PairedVolume]:
    """Yield paired (source, target) volumes for a given specialization.

    Parameters
    ----------
    records:
        Raw record list as returned by :func:`load_manifest_records`.
    data_root:
        Root directory under which ``relative_path`` values resolve.
        Corresponds to the ``ChallengeData/`` tree on the cluster.
    source_field:
        Field strength (float, e.g. ``0.1``) of the source domain.
    target_field:
        Field strength (float, e.g. ``7.0``) of the target domain.
    contrast:
        Manifest contrast key: ``"T1w"``, ``"T2w"``, or ``"T2FLAIR"``.
    split:
        Only records whose ``relative_path`` starts with this prefix are
        considered.  Defaults to ``"Validating_prospective"``.
    pairing:
        How source and target records are matched into pairs.

        - ``"ordinal"`` (default): pair by **rank within field**.  The source-field
          records (sorted by ``subject_id``) are zipped rank-for-rank with the
          target-field records (also sorted by ``subject_id``).  This reconstructs
          the travelling-volunteer correspondence of the MRIxFields2026 validation
          manifest, whose ``subject_id`` numbering is **per-field** (e.g. 0.1T =
          subjects 0001-0003, 7T = subjects 0016-0018), so ``subject_id`` never
          recurs across fields and matching by id would yield **zero** pairs.  The
          load-bearing assumption is that the *k*-th subject at each field is the
          same anatomy (which holds for the travelling-volunteer cohort).  The
          yielded ``PairedVolume.subject_id`` is ``f"{src_id}->{tgt_id}"`` so the
          pairing is visible in provenance.  When the two field lists differ in
          length, ``min(len)`` pairs are emitted and a ``WARNING`` names the counts
          (never a silent whole-drop).
        - ``"subject_id"``: pair records that share a ``subject_id`` (retrospective
          / recurring-id data).  Subjects missing either side are skipped with a
          logged ``WARNING``.

        An unknown value raises :class:`ValueError`.

    Yields
    ------
    PairedVolume
        One entry per emitted pair.  If either field list is empty, nothing is
        yielded (the zero-subjects guard in the runner surfaces the all-empty case).
    """
    if pairing not in ("ordinal", "subject_id"):
        raise ValueError(f"unknown pairing {pairing!r}; expected 'ordinal' or 'subject_id'")

    data_root = Path(data_root)
    sel = [
        r
        for r in records
        if r["contrast"] == contrast and r["relative_path"].startswith(split + "/")
    ]

    if pairing == "subject_id":
        # Index by subject → field_strength → record; pair records that share an id.
        by_subject: dict[str, dict[float, dict]] = {}
        for r in sel:
            by_subject.setdefault(r["subject_id"], {})[float(r["field_strength"])] = r

        for subject in sorted(by_subject):
            fields = by_subject[subject]
            src_r = fields.get(float(source_field))
            tgt_r = fields.get(float(target_field))
            if src_r is None or tgt_r is None:
                logger.warning(
                    "subject %s missing %sT or %sT for %s - skipping",
                    subject,
                    source_field,
                    target_field,
                    contrast,
                )
                continue
            yield _emit_pair(data_root, src_r, tgt_r, subject)
        return

    # -- ordinal: rank-within-field pairing (default) ------------------------
    src_records = sorted(
        (r for r in sel if float(r["field_strength"]) == float(source_field)),
        key=lambda r: str(r["subject_id"]),
    )
    tgt_records = sorted(
        (r for r in sel if float(r["field_strength"]) == float(target_field)),
        key=lambda r: str(r["subject_id"]),
    )
    if not src_records or not tgt_records:
        # Nothing to pair; the runner's zero-subjects guard fails loud if this
        # is true for every task (never a silent all-zero score, C2).
        return
    if len(src_records) != len(tgt_records):
        logger.warning(
            "ordinal pairing count mismatch for %s %sT->%sT: %d source vs %d target "
            "record(s); pairing the first %d by rank",
            contrast,
            source_field,
            target_field,
            len(src_records),
            len(tgt_records),
            min(len(src_records), len(tgt_records)),
        )
    # strict=False: unequal lengths are intentional (min(len) paired, warned above).
    for src_r, tgt_r in zip(src_records, tgt_records, strict=False):
        subject_id = f"{src_r['subject_id']}->{tgt_r['subject_id']}"
        yield _emit_pair(data_root, src_r, tgt_r, subject_id)
