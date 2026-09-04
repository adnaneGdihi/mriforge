"""Volume-level fMRI dataset and its NIfTI/npy readers.

Split out of :mod:`fmri_dataset` in the Wave 0 exit-criterion work (#1400):
that module was 427 LOC against the 300 ceiling (NN20). It is now a facade that
re-exports this module, :mod:`fmri_bold_series_dataset` and
:mod:`cortical_surface_dataset`, so every existing import spelling still works.

This module is the root of the three-way dependency DAG -- the surface dataset
builds on :class:`FMRIVolumeDataset` and the BOLD series dataset reuses
:func:`_read_volume`, which is why the volume half moved out rather than staying
behind: a facade that both re-exports the surface dataset *and* defines its base
class would be a circular import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _read_volume(path: Path) -> np.ndarray | None:
    """Read a NIfTI/NumPy volume, or ``None`` on failure.

    Module-level because two datasets need it and it never used ``self``.
    """
    try:
        if path.suffix == ".npy":
            return np.load(path)
        import nibabel as nib  # type: ignore

        return nib.load(str(path)).get_fdata().astype("float32")
    except Exception as exc:
        logger.warning("fmri_dataset: failed to load %s: %s", path, exc)
        return None


def _list_volumes(root: Path, exts=(".nii", ".nii.gz", ".npy")) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for ext in exts:
        out.extend(sorted(root.glob(f"**/*{ext}")))
    return out


class FMRIVolumeDataset(Dataset):
    """4-D BOLD volume dataset.

    Args:
        root: Directory containing NIfTI or NumPy 4-D files.
        target_shape: Optional resize to ``(T, H, W, D)``. ``None``
            keeps the on-disk shape.
        tr_seconds: Repetition time annotation routed into the batch
            dict so strategies that care (e.g. HRF coupling) can read
            it.
        phase_encode_axis: Routed into the batch dict for the EPI
            distortion strategy.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        target_shape: tuple[int, int, int, int] | None = None,
        tr_seconds: float = 0.72,
        phase_encode_axis: int = -2,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.files = _list_volumes(self.root)
        self.target_shape = target_shape
        self.tr_seconds = tr_seconds
        self.phase_encode_axis = phase_encode_axis

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: Path) -> np.ndarray | None:
        return _read_volume(path)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        arr = self._load(path)
        if arr is None:
            # Fail loud: substituting an all-zeros placeholder trains the model
            # on fabricated data while the run looks healthy (pitfall #9/#16 —
            # the same zero-fill facade removed from m4raw_dataset). A corrupt or
            # unreadable volume must stop the run so it can be fixed.
            raise RuntimeError(
                f"FMRIVolumeDataset: failed to load volume '{path}'. Refusing to "
                "substitute a zeros placeholder (would train on fabricated data)."
            )
        if self.target_shape is not None:
            # Brutal centre-crop / pad to target shape; production-grade
            # cohorts override this via DataPipelineDirector transforms.
            T, H, W, D = self.target_shape
            arr = self._center_fit(arr, (T, H, W, D))
        x = torch.from_numpy(arr).float()
        # Channel-first [C, T, H, W] convention for the strategies'
        # 4-D-aware code paths; we drop the depth axis when D=1.
        if x.dim() == 4:
            x = x.permute(3, 0, 1, 2) if x.shape[-1] > 1 else x.squeeze(-1).unsqueeze(0)
        return {
            "image": x,
            "source_path": str(path),
            "tr": float(self.tr_seconds),
            "phase_encode_axis": int(self.phase_encode_axis),
        }

    @staticmethod
    def _center_fit(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        out = np.zeros(shape, dtype=arr.dtype)
        slices_src = []
        slices_dst = []
        for s_src, s_dst in zip(arr.shape, shape, strict=False):
            if s_src >= s_dst:
                start = (s_src - s_dst) // 2
                slices_src.append(slice(start, start + s_dst))
                slices_dst.append(slice(0, s_dst))
            else:
                pad = (s_dst - s_src) // 2
                slices_src.append(slice(0, s_src))
                slices_dst.append(slice(pad, pad + s_src))
        out[tuple(slices_dst)] = arr[tuple(slices_src)]
        return out


def build_fmri_index(
    data_root: str | Path, glob_pattern: str = "**/*.nii*"
) -> list[dict[str, str]]:
    """Index 4-D BOLD volumes under ``data_root``.

    One record per volume. Kept separate from :class:`FMRIVolumeDataset`'s own
    directory scan so the pipeline route can split train/val the way every other
    registered dataset does, rather than re-scanning the tree twice.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"fMRI data_root does not exist: {root}")
    files = sorted(root.glob(glob_pattern))
    if not files:
        raise ValueError(
            f"No 4-D BOLD volumes matched {glob_pattern!r} under {root}. "
            "dataset_type='fmri' requires NIfTI/NumPy volumes with a trailing "
            "time axis."
        )
    return [{"volume": str(f)} for f in files]


__all__ = ["FMRIVolumeDataset", "build_fmri_index"]
