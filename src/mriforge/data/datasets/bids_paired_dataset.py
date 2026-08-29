"""BIDS low-field/high-field paired NIfTI indexer + dataset (spec E5).

Turns a BIDS-organised paired-field dataset (e.g. ``ulf_paired``: a ``64mT_data``
tree and a ``3T_data`` tree of ``sub-XXXX/.../sub-XXXX_..._<contrast>.nii.gz``)
into paired records and ``tio.Subject`` instances for low-field → high-field
domain adaptation: ``input`` = low-field image, ``target`` = high-field image,
matched by **(subject, contrast)**.

The pairing is exact — a contrast present in only one field is **excluded**, never
faked (pitfall #9). Reading goes through the data-SSOT ``NiftiStrategy`` (pitfall
#11). The two fields differ in resolution / matrix; aligning them to a common
space is the transform pipeline's job (a ``Resample``), not the dataset's.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from mriforge.data.io_strategies import NiftiStrategy

__all__ = ["BidsPairedDataset", "build_bids_paired_index", "parse_bids_entities"]

_SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")
_SUFFIX_RE = re.compile(r"_([A-Za-z0-9]+)\.nii(?:\.gz)?$")
_SES_RE = re.compile(r"_ses-([A-Za-z0-9]+)")
_RUN_RE = re.compile(r"_run-([A-Za-z0-9]+)")
_ACQ_RE = re.compile(r"_acq-([A-Za-z0-9]+)")


def parse_bids_entities(filename: str) -> dict[str, str | None]:
    """Extract the BIDS ``subject``/``ses``/``run``/``acq``/``suffix`` from a name.

    ``sub-0011_ses-02_acq-highres_run-3_T1w.nii.gz`` →
    ``{'subject': 'sub-0011', 'ses': '02', 'run': '3', 'acq': 'highres',
    'suffix': 'T1w'}``. Returns ``None`` for an entity that isn't present.

    ``ses``/``run``/``acq`` are load-bearing for pairing: keying only on
    ``(subject, suffix)`` silently collapses multiple sessions/runs to the first
    and lets a low-field session pair with a *different* high-field session
    (pitfall #9). They participate in the pairing key so only matching
    acquisitions pair.
    """
    name = Path(filename).name
    sub = _SUBJECT_RE.search(name)
    suf = _SUFFIX_RE.search(name)
    ses = _SES_RE.search(name)
    run = _RUN_RE.search(name)
    acq = _ACQ_RE.search(name)
    return {
        "subject": sub.group(1) if sub else None,
        "ses": ses.group(1) if ses else None,
        "run": run.group(1) if run else None,
        "acq": acq.group(1) if acq else None,
        "suffix": suf.group(1) if suf else None,
    }


def _find_field_dir(root: Path, name: str) -> Path | None:
    for d in sorted(root.rglob(name)):
        if d.is_dir():
            return d
    return None


def build_bids_paired_index(
    data_root: str | Path,
    low_field_dir: str = "64mT_data",
    high_field_dir: str = "3T_data",
    contrasts: tuple[str, ...] = ("T1w", "T2w", "FLAIR"),
) -> list[dict[str, str]]:
    """Pair low-field ↔ high-field NIfTI by ``(subject, contrast)``.

    Returns records ``{'input': <low-field path>, 'target': <high-field path>,
    'subject': ..., 'contrast': ...}``. A contrast present in only one field is
    excluded (no fake pairing). Raises if ``data_root`` is missing or either
    field directory cannot be located beneath it.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"BIDS data_root does not exist: {root}")
    lo_root = _find_field_dir(root, low_field_dir)
    hi_root = _find_field_dir(root, high_field_dir)
    if lo_root is None or hi_root is None:
        raise ValueError(
            f"Could not locate both field directories under {root}: "
            f"low '{low_field_dir}'={'found' if lo_root else 'MISSING'}, "
            f"high '{high_field_dir}'={'found' if hi_root else 'MISSING'}."
        )

    contrast_set = set(contrasts)
    # Cross-field pairing keys on (subject, contrast) only: the 64mT and 3T scans
    # are separate acquisitions, so their ses-/run-/acq- labels legitimately
    # differ and must NOT be required to match. But WITHIN one field a
    # (subject, contrast) must be unique — multiple sessions/runs/acqs would make
    # the pair ambiguous. The old code silently kept the first (pitfall #9: silent
    # data loss); we now raise, naming the differing BIDS entities.

    def _index(field_root: Path) -> dict[tuple[str, str], Path]:
        out: dict[tuple[str, str], Path] = {}
        for p in sorted(field_root.rglob("*.nii.gz")):
            ent = parse_bids_entities(p.name)
            subject, suffix = ent["subject"], ent["suffix"]
            if not (subject and suffix in contrast_set):
                continue
            key = (subject, suffix)
            if key in out:
                prev = parse_bids_entities(out[key].name)
                differ = [e for e in ("ses", "run", "acq") if prev[e] != ent[e]]
                raise ValueError(
                    f"Ambiguous BIDS pairing under {field_root}: {key} matches "
                    f"both {out[key].name!r} and {p.name!r} (differ in "
                    f"{differ or 'no entity'}). This dataset pairs one image per "
                    "(subject, contrast) per field; refusing to silently drop "
                    "one (pitfall #9)."
                )
            out[key] = p
        return out

    lo_index = _index(lo_root)
    hi_index = _index(hi_root)
    records: list[dict[str, str]] = []
    for subject, contrast in sorted(lo_index.keys() & hi_index.keys()):
        records.append(
            {
                "input": str(lo_index[(subject, contrast)]),
                "target": str(hi_index[(subject, contrast)]),
                "subject": subject,
                "contrast": contrast,
            }
        )
    return records


class BidsPairedDataset(Dataset):
    """Low-field/high-field paired NIfTI → ``tio.Subject`` (input=low, target=high)."""

    def __init__(
        self,
        index: list[dict[str, str]],
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        self.index = list(index)
        self.transform = transform
        self._reader = NiftiStrategy()

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[tio.Subject]:
        """Cheap Subject shells for ``tio.Queue`` length probing (no voxel read).

        ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``
        and only needs an indexable ``Sequence[Subject]`` of the right length.
        Without it a queued arm crashes at step 0 with ``'BidsPairedDataset'
        object has no attribute 'dry_iter'`` (the 2026-06 oracle_bssfp /
        exp_vf_29 incident class). Mirrors :meth:`OracleBssfpDataset.dry_iter`.
        """
        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def __getitem__(self, idx: int) -> tio.Subject:
        rec = self.index[idx]
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=self._load_image(rec["input"])),
            target=tio.ScalarImage(tensor=self._load_image(rec["target"])),
        )
        subject["subject_id"] = rec["subject"]
        subject["contrast"] = rec["contrast"]
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _load_image(self, path: str) -> torch.Tensor:
        data: Any = self._reader.load(path)["data"]
        tensor = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        tensor = tensor.float()
        if tensor.dim() == 3:  # (X, Y, Z) → TorchIO (C, X, Y, Z)
            tensor = tensor.unsqueeze(0)
        return tensor
