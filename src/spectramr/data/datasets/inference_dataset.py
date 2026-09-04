"""Inference Dataset
=================

Dedicated dataset for batched inference operations.
Decoupled from monolith `unified_inference.py`.

Every slice of a 3-D input volume is served as its own inference sample: the
index is expanded to one ``(file, slice)`` entry per slice, so a depth-``D``
volume yields ``D`` samples rather than silently collapsing to its central
slice. An explicit ``slice_index`` pins a single slice per file (legacy
single-slice mode).

.. mermaid::

    flowchart TD
        Files[File Paths] --> Probe[Probe slice count per file]
        Probe --> Index["(file, slice) index"]
        Index -->|Iterate| Load{Load & Process one slice}

        Load -->|DICOM| DicomAdapter
        Load -->|NIfTI| NiftiAdapter
        Load -->|H5| H5Adapter
        Load -->|Image| StandardAdapter

        DicomAdapter & NiftiAdapter & H5Adapter & StandardAdapter --> Tensor[Raw Tensor]

        Tensor --> Process{Post-Process}
        Process -->|Resize| Res[Target HW]
        Process -->|Channels| Ch[Channel Fix]

        Ch --> Output[Final Tensor]
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from spectramr.domain.entities.optional_imports import has_h5py
from spectramr.infrastructure.io.adapters import (
    load_dicom,
    load_h5_kspace,
    load_image_standard,
    load_nifti,
    load_numpy,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.shared.utils.security import validate_image_file

logger = logging.getLogger(__name__)


def _is_nifti(path: Path) -> bool:
    """True for ``.nii`` / ``.nii.gz`` (the compound suffix ``path.suffix`` misses)."""
    return path.suffix.lower() == ".nii" or path.name.lower().endswith(".nii.gz")


def _is_kspace_file(path: Path) -> bool:
    return path.suffix.lower() in (".h5", ".hdf5")


class InferenceDataset(Dataset):
    """Dataset for batched inference — one sample per slice of every volume."""

    def __init__(
        self,
        file_paths: list[Path],
        gt_paths: list[Path] | None = None,
        slice_index: int | None = None,
        is_kspace_model: bool = False,
        target_hw: tuple[int, int] = (128, 128),
        transform: Any | None = None,
        cache_size: int = 2,
    ):
        """__init__.

        Args:
            file_paths: Input file paths (2-D or 3-D volumes).
            gt_paths: Optional per-file ground-truth paths (sliced in lockstep).
            slice_index: When set, pin this single slice for every file (legacy
                one-sample-per-file mode). When ``None`` (default) EVERY slice of
                every volume is served as its own sample.
            is_kspace_model: Whether the model consumes 2-channel k-space.
            target_hw: Target ``(H, W)`` for resizing.
            transform: Optional per-sample transform.
            cache_size: Number of recently-decoded full volumes to retain (LRU),
                so sequential per-slice access does not re-decode the file for
                every slice.
        """
        self.files = file_paths
        self.gt_files = gt_paths
        self.slice_index = slice_index
        self.is_kspace_model = is_kspace_model
        self.target_hw = target_hw
        self.transform = transform

        self._cache_size = max(1, int(cache_size))
        self._raw_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # Expand to one (file_idx, slice_idx) sample per slice. An explicit
        # slice_index pins a single slice per file; otherwise every slice of
        # every volume is served rather than silently dropping to the middle.
        self._index: list[tuple[int, int]] = []
        if slice_index is not None:
            self._index = [(fi, slice_index) for fi in range(len(file_paths))]
        else:
            for fi, path in enumerate(file_paths):
                n_slices = self._probe_num_slices(Path(path))
                self._index.extend((fi, si) for si in range(n_slices))

        # Per-file slice counts so a unique key is only appended when a file
        # actually contributes more than one slice.
        self._file_counts: dict[int, int] = {}
        for fi, _ in self._index:
            self._file_counts[fi] = self._file_counts.get(fi, 0) + 1

    def __len__(self) -> int:
        """Number of (file, slice) samples served."""
        return len(self._index)

    # ------------------------------------------------------------------ probing
    def _probe_num_slices(self, path: Path) -> int:
        """Resolve a file's slice count (cheap header read where possible).

        Indexing runs over EVERY input file at construction, so a probe that
        decodes is paid once per file before a single sample is served, and then
        largely paid again at serve time — the raw LRU holds only
        ``cache_size`` volumes, so on any corpus larger than the cache the
        probe's decode is evicted before it is used (#393).

        NIfTI, HDF5 and ``.npy`` all publish their shape in the header, so the
        decode is avoidable for them. Formats that genuinely cannot answer
        without decoding fall through to the cached full read, and say so at
        debug level rather than pretending the probe was cheap.
        """
        if _is_nifti(path):
            # Lazy header read — no voxel decode.
            from spectramr.data.io_strategies import lazy_load_nifti

            shape = tuple(int(s) for s in lazy_load_nifti(str(path)).shape)
            return shape[0] if len(shape) >= 3 else 1

        rank = 4 if _is_kspace_file(path) else 3
        shape = self._probe_shape_no_decode(path)
        if shape is not None:
            return int(shape[0]) if len(shape) == rank else 1

        logger.debug(
            "[InferenceDataset] %s has no header shape probe; decoding to count slices.",
            path.name,
        )
        raw = self._raw_full(path)
        return int(raw.shape[0]) if raw.ndim == rank else 1

    @staticmethod
    def _probe_shape_no_decode(path: Path) -> tuple[int, ...] | None:
        """Array shape from the file header, or ``None`` when unavailable.

        Returns ``None`` — never a guess — for formats without a readable
        header, so the caller falls back to a real decode rather than serving a
        wrong slice count.
        """
        suffix = path.suffix.lower()
        try:
            if suffix in (".h5", ".hdf5") and has_h5py():
                import h5py

                with h5py.File(str(path), "r") as f:
                    key = next(
                        (k for k in ("kspace", "reconstruction_rss") if k in f),
                        None,
                    )
                    if key is None:
                        return None
                    return tuple(int(d) for d in f[key].shape)

            if suffix == ".npy":
                # numpy's own header parser: reads the magic + header dict and
                # stops before the data block.
                from numpy.lib.format import read_array_header_1_0, read_magic

                with open(path, "rb") as fh:
                    read_magic(fh)
                    shape, _fortran, _dtype = read_array_header_1_0(fh)
                    return tuple(int(d) for d in shape)
        except Exception as exc:  # a probe must never be fatal
            logger.debug(
                "[InferenceDataset] header probe failed for %s (%s); falling back to a decode.",
                path.name,
                exc,
            )
        return None

    # ------------------------------------------------------------------ decoding
    def _raw_full(self, path: Path) -> np.ndarray:
        """Decode the FULL volume (all slices), with a small per-file LRU cache."""
        key = str(path)
        cached = self._raw_cache.get(key)
        if cached is not None:
            self._raw_cache.move_to_end(key)
            return cached
        raw = self._decode_full(path)
        self._raw_cache[key] = raw
        while len(self._raw_cache) > self._cache_size:
            self._raw_cache.popitem(last=False)
        return raw

    def _decode_full(self, path: Path) -> np.ndarray:
        """Load every slice via the format adapter (``slice_index=None``)."""
        validated_path = validate_image_file(path)
        suffix = validated_path.suffix.lower()

        if suffix in (".dcm", ".dicom"):
            return load_dicom(validated_path, None)
        if _is_nifti(validated_path):
            return load_nifti(validated_path, None)
        if suffix in (".h5", ".hdf5") and has_h5py():
            return load_h5_kspace(validated_path, None)
        if suffix in (".npy", ".npz"):
            return load_numpy(validated_path)
        return load_image_standard(validated_path)

    # ------------------------------------------------------------------ per-slice
    def _resize_hw(self, image: np.ndarray) -> np.ndarray:
        """Resize the leading ``(H, W)`` plane to ``target_hw`` if needed."""
        if image.shape[:2] == tuple(self.target_hw):
            return image
        try:
            from scipy import ndimage

            zoom = [
                self.target_hw[0] / image.shape[0],
                self.target_hw[1] / image.shape[1],
            ]
            zoom += [1.0] * (image.ndim - 2)
            return ndimage.zoom(image, zoom, order=1)
        except ImportError:
            return image

    def _process(self, path: Path, slice_idx: int) -> torch.Tensor:
        """Load one slice of ``path`` and shape it for the model."""
        raw = self._raw_full(path)

        if _is_kspace_file(path):
            # raw is [S, H, W, 2] (multi-slice) or [H, W, 2] (single slice).
            plane = raw[slice_idx] if raw.ndim == 4 else raw  # [H, W, 2]
            plane = self._resize_hw(plane)
            return torch.from_numpy(plane).permute(2, 0, 1).float().clone()  # [2, H, W]

        # Image domain: raw is [S, H, W] (volume) or [H, W] (single slice).
        image = raw[slice_idx] if raw.ndim == 3 else raw
        image = self._resize_hw(image)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)  # [1, H, W]

        # Enforce channel consistency.
        if self.is_kspace_model:
            if tensor.shape[0] == 1:
                kspace_complex = fft2c(tensor.unsqueeze(0))  # [1, 1, H, W]
                tensor = torch.cat([kspace_complex.real, kspace_complex.imag], dim=1).squeeze(
                    0
                )  # [2, H, W]
        else:
            if tensor.shape[0] == 2:
                real_part, imag_part = tensor[0], tensor[1]
                tensor = torch.sqrt(real_part**2 + imag_part**2 + 1e-12).unsqueeze(0)
            elif tensor.shape[0] == 3:
                tensor = (0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]).unsqueeze(0)

        return tensor.clone()

    def _sample_key(self, file_path: Path, file_idx: int, slice_idx: int) -> str:
        """Unique key per sample so per-slice outputs never overwrite.

        A bare path is kept when the file contributes a single sample (2-D input
        or an explicit ``slice_index``); otherwise the slice index is appended.
        """
        if self.slice_index is not None or self._file_counts.get(file_idx, 1) <= 1:
            return str(file_path)
        return f"{file_path}#slice{slice_idx:04d}"

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, torch.Tensor, bool]:
        """Return ``(key, input, gt, has_gt)`` for one (file, slice) sample."""
        file_idx, slice_idx = self._index[idx]
        file_path = Path(self.files[file_idx])
        try:
            input_tensor = self._process(file_path, slice_idx)

            if self.transform:
                input_tensor = self.transform(input_tensor)

            gt_tensor = torch.zeros_like(input_tensor)
            has_gt = False
            if self.gt_files and self.gt_files[file_idx]:
                try:
                    gt_tensor = self._process(Path(self.gt_files[file_idx]), slice_idx)
                    if self.transform:
                        gt_tensor = self.transform(gt_tensor)

                    if gt_tensor.shape != input_tensor.shape:
                        gt_tensor = torch.nn.functional.interpolate(
                            gt_tensor.unsqueeze(0),
                            size=input_tensor.shape[1:],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    has_gt = True
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

            key = self._sample_key(file_path, file_idx, slice_idx)
            return key, input_tensor, gt_tensor, has_gt

        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}") from e
