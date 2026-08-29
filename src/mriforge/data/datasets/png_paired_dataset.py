"""Paired-PNG super-resolution dataset (spec — brats_sr).

Pairs a low-resolution / undersampled PNG directory (``A_LRSI``) against a
high-resolution / fully-sampled one (``A_HRSI``) by **common filename** for
super-resolution / reconstruction training: ``input`` = LR, ``target`` = HR, with
an optional ``Lesion_map`` segmentation carried as a ``tio.LabelMap``.

This *reuses* the PNG helpers in ``data/datasets/utils/image_volume_utils.py``
(``find_common_png_stems`` / ``load_grayscale_png``) that previously had no
pipeline consumer — wiring them in fixes the latent inert-mechanism gap
(pitfall #16). A filename present in only one field is excluded (no fake
pairing — pitfall #9).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torchio as tio
from torch.utils.data import Dataset

from mriforge.data.datasets.utils.image_volume_utils import (
    find_common_png_stems,
    list_png_files,
    load_grayscale_png,
)

__all__ = ["PngPairedDataset", "build_png_paired_index"]


def _find_dir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    for d in sorted(root.rglob(name)):
        if d.is_dir():
            return d
    return None


def build_png_paired_index(
    data_root: str | Path,
    lr_dir: str = "A_LRSI",
    hr_dir: str = "A_HRSI",
    lesion_dir: str | None = None,
) -> list[dict[str, str | None]]:
    """Pair LR ↔ HR PNGs by common filename.

    Returns records ``{'input': <LR path>, 'target': <HR path>,
    'lesion': <path|None>, 'stem': <filename>}``. A filename present in only one
    of LR/HR is excluded. Raises if ``data_root`` is missing or either field
    directory cannot be located beneath it.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"PNG-paired data_root does not exist: {root}")
    lr = _find_dir(root, lr_dir)
    hr = _find_dir(root, hr_dir)
    if lr is None or hr is None:
        raise ValueError(
            f"Could not locate both PNG directories under {root}: "
            f"lr '{lr_dir}'={'found' if lr else 'MISSING'}, "
            f"hr '{hr_dir}'={'found' if hr else 'MISSING'}."
        )
    lesion = _find_dir(root, lesion_dir) if lesion_dir else None
    lesion_names = {p.name for p in list_png_files(lesion)} if lesion else set()

    records: list[dict[str, str | None]] = []
    for name in find_common_png_stems([lr, hr]):
        lesion_path = str(lesion / name) if (lesion is not None and name in lesion_names) else None
        records.append(
            {
                "input": str(lr / name),
                "target": str(hr / name),
                "lesion": lesion_path,
                "stem": name,
            }
        )
    return records


class PngPairedDataset(Dataset):
    """LR/HR PNG pairs → ``tio.Subject`` (input=LR, target=HR, optional lesion)."""

    def __init__(
        self,
        index: list[dict[str, str | None]],
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        self.index = list(index)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[tio.Subject]:
        """Cheap Subject shells for ``tio.Queue`` length probing (no voxel read).

        ``tio.Queue.iterations_per_epoch`` calls ``subjects_dataset.dry_iter()``
        and only needs an indexable ``Sequence[Subject]`` of the right length.
        Without it a queued arm crashes at step 0 with ``'PngPairedDataset'
        object has no attribute 'dry_iter'`` (the 2026-06 oracle_bssfp /
        exp_vf_29 incident class). Mirrors :meth:`OracleBssfpDataset.dry_iter`.
        """
        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def __getitem__(self, idx: int) -> tio.Subject:
        rec = self.index[idx]
        # load_grayscale_png → (1, H, W); TorchIO wants (C, X, Y, Z) → add depth.
        kwargs: dict[str, tio.Image] = {
            "input": tio.ScalarImage(
                tensor=load_grayscale_png(Path(str(rec["input"]))).unsqueeze(-1)
            ),
            "target": tio.ScalarImage(
                tensor=load_grayscale_png(Path(str(rec["target"]))).unsqueeze(-1)
            ),
        }
        lesion = rec.get("lesion")
        if lesion:
            kwargs["lesion"] = tio.LabelMap(
                tensor=load_grayscale_png(Path(str(lesion))).unsqueeze(-1)
            )
        subject = tio.Subject(**kwargs)
        subject["stem"] = rec["stem"]
        if self.transform is not None:
            subject = self.transform(subject)
        return subject
