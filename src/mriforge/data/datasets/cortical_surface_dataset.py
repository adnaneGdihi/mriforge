"""Cortical-surface dataset and its GIFTI/CIFTI/FreeSurfer readers.

Split out of :mod:`fmri_dataset` in the Wave 0 exit-criterion work (#1400).
Reachable under its original spelling -- ``fmri_dataset`` re-exports it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mriforge.data.datasets.fmri_volume_dataset import FMRIVolumeDataset

logger = logging.getLogger(__name__)


def _load_gifti_array(path: Path) -> np.ndarray | None:
    """Read a GIFTI ``.gii`` file via nibabel; return the first
    DataArray flattened to a NumPy array, or ``None`` on failure."""
    try:
        import nibabel as nib  # type: ignore

        gii = nib.load(str(path))
        if not gii.darrays:
            return None
        return np.asarray(gii.darrays[0].data, dtype="float32")
    except Exception as exc:
        logger.warning("GIFTI read failed for %s: %s", path, exc)
        return None


def _load_cifti_array(path: Path) -> np.ndarray | None:
    """Read a CIFTI ``.dscalar.nii`` / ``.dtseries.nii`` via nibabel."""
    try:
        import nibabel as nib  # type: ignore

        c = nib.load(str(path))
        return np.asarray(c.get_fdata(), dtype="float32")
    except Exception as exc:
        logger.warning("CIFTI read failed for %s: %s", path, exc)
        return None


def _load_freesurfer_curv(path: Path) -> np.ndarray | None:
    """Read a FreeSurfer ``.curv`` / ``.thickness`` / ``.sulc`` file."""
    try:
        import nibabel as nib  # type: ignore

        return np.asarray(nib.freesurfer.read_morph_data(str(path)), dtype="float32")
    except Exception as exc:
        logger.warning("FreeSurfer read failed for %s: %s", path, exc)
        return None


class CorticalSurfaceDataset(Dataset):
    """BOLD volumes paired with a precomputed conformal flattening grid.

    Surface companion ingestion order (per sample):

    1. ``X_cortex_flatten.npy`` (legacy / unit-test path).
    2. ``X_cortex_flatten.gii`` (GIFTI).
    3. ``X_cortex_flatten.dscalar.nii`` (CIFTI).
    4. ``X_cortex_flatten.curv`` (FreeSurfer morph).

    The first match wins and is stored on the batch under
    ``"_cortex_flatten_grid_override"``. The downstream strategy
    consumes it via the
    :func:`mriforge.data.transforms.sfc_conformal_fmri_keys.attach_cortex_flatten_grid`
    transform.
    """

    SURFACE_SUFFIXES = (
        ("_cortex_flatten.npy", _load_freesurfer_curv.__class__),  # marker only
        ("_cortex_flatten.gii", None),
        ("_cortex_flatten.dscalar.nii", None),
        ("_cortex_flatten.curv", None),
    )

    def __init__(
        self,
        root: str | Path,
        *,
        grid_suffix: str = "_cortex_flatten.npy",
    ) -> None:
        super().__init__()
        self.volume_ds = FMRIVolumeDataset(root)
        self.grid_suffix = grid_suffix

    def __len__(self) -> int:
        return len(self.volume_ds)

    def _try_load_grid(self, base: Path) -> np.ndarray | None:
        """Try each known surface companion suffix in priority order."""
        npy_path = Path(str(base) + self.grid_suffix)
        if npy_path.exists():
            try:
                return np.load(npy_path).astype("float32")
            except Exception as exc:
                logger.warning("npy surface read failed for %s: %s", npy_path, exc)
        for suffix, _ in self.SURFACE_SUFFIXES[1:]:
            candidate = Path(str(base) + suffix)
            if not candidate.exists():
                continue
            if suffix.endswith(".gii"):
                arr = _load_gifti_array(candidate)
            elif suffix.endswith(".dscalar.nii") or suffix.endswith(".dtseries.nii"):
                arr = _load_cifti_array(candidate)
            elif suffix.endswith(".curv"):
                arr = _load_freesurfer_curv(candidate)
            else:
                arr = None
            if arr is not None:
                return arr
        return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.volume_ds[idx]
        path = Path(sample["source_path"])
        base = (
            path.with_suffix("").with_suffix("")
            if path.name.endswith(".nii.gz")
            else path.with_suffix("")
        )
        arr = self._try_load_grid(base)
        if arr is not None:
            sample["_cortex_flatten_grid_override"] = torch.from_numpy(arr)
        return sample


__all__ = ["CorticalSurfaceDataset"]
