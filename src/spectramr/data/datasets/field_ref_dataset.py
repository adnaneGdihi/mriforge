"""NIfTI field-reference dataset (spec — kasper / traveling_heads).

Pairs each anatomy NIfTI (the VF target) with a real field map carried as
``b0_map`` / ``b1_map`` for the real-reference field-scoring seam. This is the
*derivation-free* counterpart to ``BartKspaceDataset``'s ``b0_from_echo`` /
``b1_from_dam``: when a dataset ships the field map directly (e.g. kasper's
``b0MapSmoothed..._Hz.nii`` in Hz, or traveling_heads' B1+ maps), no echo /
double-angle inversion is needed — the map is loaded as-is through the SSOT
``NiftiStrategy``.

The field map is applied to *every* anatomy in the directory (the VF twin applies
it as an independent corruption — the field's spatial structure is what matters,
not an anatomy↔field correspondence).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.data.io_strategies import NiftiStrategy

__all__ = ["FieldRefDataset", "build_field_ref_index"]


def _resolve(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else root / pp


def build_field_ref_index(
    data_root: str | Path,
    anatomy_glob: str,
    b0_map: str | None = None,
    b1_map: str | None = None,
) -> list[dict[str, str | None]]:
    """Pair each anatomy NIfTI (``anatomy_glob`` under ``data_root``) with the
    shared field map(s).

    Returns records ``{'target', 'b0_map', 'b1_map'}``. Raises if ``data_root`` is
    missing, neither field map is given, a named field map is absent, or the glob
    matches no anatomy.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"field_ref data_root does not exist: {root}")
    if b0_map is None and b1_map is None:
        raise ValueError(
            "field_ref requires at least one of b0_map / b1_map (the real field "
            "the run grades against)."
        )
    b0 = _resolve(root, b0_map) if b0_map else None
    b1 = _resolve(root, b1_map) if b1_map else None
    for name, pth in (("b0_map", b0), ("b1_map", b1)):
        if pth is not None and not pth.is_file():
            raise FileNotFoundError(f"field_ref {name} not found: {pth}")

    anatomy = sorted(root.glob(anatomy_glob))
    if not anatomy:
        raise FileNotFoundError(f"field_ref found no anatomy under {root}/{anatomy_glob}")
    return [
        {
            "target": str(a),
            "b0_map": str(b0) if b0 else None,
            "b1_map": str(b1) if b1 else None,
        }
        for a in anatomy
    ]


class FieldRefDataset(Dataset):
    """Anatomy NIfTI → ``tio.Subject`` (input/target = anatomy, + real b0_map/b1_map)."""

    def __init__(
        self,
        index: list[dict[str, str | None]],
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
        Without it a queued arm crashes at step 0 with ``'FieldRefDataset'
        object has no attribute 'dry_iter'`` (the 2026-06 oracle_bssfp /
        exp_vf_29 incident class). Mirrors :meth:`OracleBssfpDataset.dry_iter`.
        """
        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def __getitem__(self, idx: int) -> tio.Subject:
        rec = self.index[idx]
        anatomy = self._load(str(rec["target"]))
        kwargs: dict[str, tio.Image] = {
            "input": tio.ScalarImage(tensor=anatomy),  # twin degrades downstream
            "target": tio.ScalarImage(tensor=anatomy.clone()),
        }
        if rec.get("b0_map"):
            kwargs["b0_map"] = tio.ScalarImage(tensor=self._load(str(rec["b0_map"])))
        if rec.get("b1_map"):
            kwargs["b1_map"] = tio.ScalarImage(tensor=self._load(str(rec["b1_map"])))
        subject = tio.Subject(**kwargs)
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _load(self, path: str) -> torch.Tensor:
        data: Any = self._reader.load(path)["data"]
        tensor = data if torch.is_tensor(data) else torch.as_tensor(np.asarray(data))
        tensor = tensor.float()
        if tensor.dim() == 3:  # (X, Y, Z) → TorchIO (C, X, Y, Z)
            tensor = tensor.unsqueeze(0)
        return tensor
