"""4D cine MRI dataset (Phase 4c).

Loads 4D cardiac cine volumes shaped ``[H, W, slices, frames]`` and folds
the frame axis into TorchIO's channel slot so the spatial sampling /
transform machinery works unchanged.

Convention
==========

On-disk tensor shape (per config ``temporal.frame_axis``)::

    ACDC NIfTI 4D : (H, W, slices, frames)  → frame_axis=3 (default)
    FastMRI cine  : (frames, H, W, slices)  → frame_axis=0

After loading, the dataset permutes to TorchIO channel-first convention::

    tio.ScalarImage shape: (frames, W, H, slices)
                              ↑
                              C slot — temporal samplers slice here

The frame ordering is preserved through the permute and recorded on the
Subject as ``subject["frame_order"]`` so the temporal grid aggregator
(Phase 4c) can reassemble cine loops correctly at inference.

Format support: NIfTI (via nibabel), .npy, .pt. ACDC is the canonical
NIfTI 4D layout.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.config.schemas.data import TemporalConfigSchema

__all__ = ["CineMRIDataset", "build_cine_index"]

# Volume extensions this module can load, longest-first so the compound
# ``.nii.gz`` wins over ``.gz``. One vocabulary for both the loader and the
# sibling-name rule -- two lists would let ``build_cine_index`` pair a file
# ``_load_4d_volume`` then refuses.
_VOLUME_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".npy", ".pt")


def _volume_stem(path: Path) -> str:
    """Return the filename with its volume extension removed.

    ``target_suffix`` is substituted for this extension, so the rule has to
    know the whole vocabulary. The previous ``name.replace(".nii.gz", suffix)``
    was a silent no-op on any other format -- and a no-op leaves
    ``target_path == path``, which pairs a volume with **itself**. That was
    unreachable only because ``glob_pattern`` was hardcoded to NIfTI.
    """
    lowered = path.name.lower()
    for suffix in _VOLUME_SUFFIXES:
        if lowered.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(
        f"Cannot derive a sibling name for {path.name}: it ends with none of "
        f"{list(_VOLUME_SUFFIXES)}. Narrow temporal.glob_pattern, or drop "
        "temporal.target_suffix and self-pair."
    )


def _load_4d_volume(path: Path) -> torch.Tensor:
    """Load a 4D volume from NIfTI / NumPy / PyTorch file."""
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        try:
            import nibabel as nib  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "CineMRIDataset requires nibabel for NIfTI loading. "
                "Install with `pip install nibabel`."
            ) from e
        arr = nib.load(str(path)).get_fdata()  # type: ignore[no-untyped-call]
        tensor = torch.as_tensor(arr, dtype=torch.float32)
    elif suffix.endswith(".npy"):
        import numpy as np

        tensor = torch.from_numpy(np.load(str(path))).to(torch.float32)
    elif suffix.endswith(".pt"):
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        tensor = (
            loaded.float()
            if isinstance(loaded, torch.Tensor)
            else torch.as_tensor(loaded, dtype=torch.float32)
        )
    else:
        raise ValueError(
            f"Unsupported cine volume format: {path}. Use .nii / .nii.gz / .npy / .pt."
        )
    if tensor.dim() != 4:
        raise ValueError(
            f"Expected 4D cine volume from {path}, got {tensor.dim()}D "
            f"(shape={tuple(tensor.shape)})."
        )
    return tensor


def _frame_axis_to_channel(tensor: torch.Tensor, frame_axis: int) -> torch.Tensor:
    """Move the frame axis to position 0 (TorchIO channel slot).

    The remaining three spatial axes become (W, H, D) in their original
    relative order — this matches TorchIO's spatial-dim convention.
    """
    if frame_axis < 0 or frame_axis > 3:
        raise ValueError(f"frame_axis must be in [0, 3]; got {frame_axis}.")
    # Build permutation: target dim 0 is frame_axis, others keep order.
    src_dims = [frame_axis] + [i for i in range(4) if i != frame_axis]
    return tensor.permute(*src_dims).contiguous()


class CineMRIDataset(Dataset):
    """Yields TorchIO Subjects for 4D cine reconstruction.

    Args:
        index: List of records. Each record at minimum has ``"path"``
            (4D volume) and optionally ``"target_path"`` (paired target
            for translation/super-resolution).
        temporal_config: Populated :class:`TemporalConfigSchema`.
        transform: Optional TorchIO transform pipeline.

    Per-Subject output::

        tio.Subject {
            input:        ScalarImage((frames, W, H, slices)),
            target:       ScalarImage(...) — always present, see below
            frame_order:  list[int]   — original frame indices [0, 1, ..., F-1]
            n_frames:     int          — total frames in the volume
            subject_id:   str,
            file_id:      str,
        }

    ``target`` is mandatory: :meth:`BatchAdapter.from_dict` raises without
    it, so a Subject that omits it cannot reach a strategy. Its source is
    whichever ``temporal_config.target_source`` declares -- ``sibling``
    loads ``record["target_path"]``, ``self`` clones the input for a
    downstream degradation transform to corrupt. The dataset never infers
    which; the schema requires the arm to say.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        temporal_config: TemporalConfigSchema,
        transform: Callable | None = None,
    ):
        if not temporal_config.enabled:
            raise ValueError("CineMRIDataset requires temporal_config.enabled=True.")
        self.index = index
        self.config = temporal_config
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tio.Subject:
        record = self.index[idx]
        path = Path(record["path"])
        raw = _load_4d_volume(path)

        if self.config.total_frames is not None:
            actual_frames = raw.shape[self.config.frame_axis]
            if actual_frames != self.config.total_frames:
                raise ValueError(
                    f"Cine volume {path} has {actual_frames} frames along "
                    f"axis {self.config.frame_axis}, but "
                    f"temporal.total_frames={self.config.total_frames}. "
                    "Set temporal.total_frames=null to allow heterogeneous-"
                    "frame-count cohorts."
                )

        input_tensor = _frame_axis_to_channel(raw, self.config.frame_axis)
        n_frames = input_tensor.shape[0]

        kwargs: dict[str, Any] = {
            "input": tio.ScalarImage(tensor=input_tensor),
            "frame_order": list(range(n_frames)),
            "n_frames": n_frames,
            "subject_id": record.get("subject_id", f"cine-{idx:06d}"),
            "file_id": record.get("file_id", path.stem),
        }

        kwargs["target"] = self._build_target(record, input_tensor)

        subject = tio.Subject(**kwargs)
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _build_target(self, record: dict[str, Any], input_tensor: torch.Tensor) -> tio.ScalarImage:
        """Resolve the Subject's ``target`` per the declared ``target_source``.

        Kept off ``__getitem__`` so the two modes read side by side, and so the
        ``sibling``-without-a-path case is a raise rather than a missing key
        surfacing 200 lines later as ``BatchAdapter requires canonical keys``.
        """
        target_path = record.get("target_path")
        if self.config.target_source == "sibling":
            if target_path is None:
                raise ValueError(
                    f"temporal.target_source='sibling' but the index record for "
                    f"{record.get('path')!r} carries no 'target_path'. "
                    "build_cine_index raises on a missing sibling, so an index "
                    "built any other way is the likely source."
                )
            target_raw = _load_4d_volume(Path(target_path))
            return tio.ScalarImage(
                tensor=_frame_axis_to_channel(target_raw, self.config.frame_axis)
            )
        # 'self': the loaded volume IS the target; a degradation transform
        # corrupts ``input`` downstream. Mirrors FieldRefDataset's twin pair.
        return tio.ScalarImage(tensor=input_tensor.clone())

    def dry_iter(self) -> list[tio.Subject]:
        """Subject shells for TorchIO Queue init (no voxels loaded)."""
        stub = torch.zeros(1, 1, 1, 1)
        out: list[tio.Subject] = []
        for record in self.index:
            # ``target`` is unconditional here because __getitem__ emits it
            # unconditionally. A shell whose key set differs from the real
            # Subject's is what the Queue sizes itself against.
            kwargs: dict[str, Any] = {
                "input": tio.ScalarImage(tensor=stub),
                "target": tio.ScalarImage(tensor=stub),
                "frame_order": [0],
                "n_frames": 1,
                "subject_id": record.get("subject_id", "unknown"),
                "file_id": record.get("file_id", "unknown"),
            }
            out.append(tio.Subject(**kwargs))
        return out


def build_cine_index(
    data_root: str | Path,
    glob_pattern: str = "**/*4d.nii.gz",
    target_suffix: str | None = None,
) -> list[dict[str, Any]]:
    """Glob a directory for cine 4D volumes.

    Args:
        data_root: Top-level directory to search.
        glob_pattern: Pattern matching cine inputs. Default matches
            ACDC-style ``*4d.nii.gz`` files.
        target_suffix: If set, every matched input must have a sibling
            named by substituting this for its volume extension
            (e.g. ``"_gt.nii.gz"`` beside an input ``"*_lr.nii.gz"``).

    Returns:
        List of records suitable for :class:`CineMRIDataset`.

    Raises:
        FileNotFoundError: If ``target_suffix`` is set and any matched
            input has no sibling. Reported as one list, not one raise per
            file -- a 100-subject cohort with a typo'd suffix should cost
            one run, not one hundred.
    """
    root = Path(data_root)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(root.glob(glob_pattern)):
        if not path.is_file():
            continue
        record: dict[str, Any] = {
            "path": str(path),
            "subject_id": path.parent.name,
            "file_id": path.stem,
        }
        if target_suffix is not None:
            target_path = path.parent / (_volume_stem(path) + target_suffix)
            if target_path.exists():
                record["target_path"] = str(target_path)
            else:
                missing.append(f"{path.name} -> {target_path.name}")
        records.append(record)

    if missing:
        shown = missing[:10]
        more = f" (+{len(missing) - len(shown)} more)" if len(missing) > len(shown) else ""
        raise FileNotFoundError(
            f"target_suffix={target_suffix!r} was requested but "
            f"{len(missing)} of {len(records)} cine inputs under {root} have "
            f"no sibling{more}:\n  " + "\n  ".join(shown) + "\n"
            "Every input needs its pair -- silently dropping the target here "
            "is how an arm reaches the trainer with an input and no target."
        )
    return records
