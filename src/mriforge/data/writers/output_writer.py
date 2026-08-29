"""Inference output writer (Phase 4d).

Single entry point for all on-disk inference writes:

- Splits multi-channel outputs into per-channel files with per-map
  scale/offset/dtype (Phase 4b's qMRI use case).
- Restores frame order for cine 4D outputs (Phase 4c's cine use case).
- Picks the right backend (NIfTI / H5 / NPY) from the YAML.

Why a centralized writer?
=========================

Before Phase 4d, ad-hoc ``np.save`` / ``nib.save`` calls were scattered
across inference scripts. Each call:

- guessed the channel order (was channel 0 T1 or PD? depended on the
  YAML — and the inference script had to re-read the YAML to know)
- applied a single global rescale (clipped T1 vs PD)
- ignored frame order for cine outputs (broke loop playback)

This module folds all three concerns into one config-driven path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mriforge.config.schemas.data import (
    ModeOutputSchema,
    OutputChannelSchema,
)

logger = logging.getLogger(__name__)


__all__ = ["OutputWriter", "write_output"]


# Mapping from schema dtype → numpy dtype.
_DTYPE_MAP: dict[str, np.dtype] = {
    "float32": np.dtype(np.float32),
    "float16": np.dtype(np.float16),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
}


def _apply_channel_transform(
    channel_tensor: np.ndarray, channel: OutputChannelSchema
) -> np.ndarray:
    """Apply (scale, offset, dtype) to one channel.

    For integer dtypes, rounds + clips to the dtype range — caller is
    responsible for picking scale/offset that keep values in-range.
    """
    out = channel_tensor.astype(np.float64) * channel.scale + channel.offset
    np_dtype = _DTYPE_MAP[channel.dtype]
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(np_dtype)


def _to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize input to numpy, detaching from the autograd graph."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _permute_to_frame_axis(volume: np.ndarray, source_axis: int, target_axis: int) -> np.ndarray:
    """Move the frame axis from source position to target position."""
    if source_axis == target_axis:
        return volume
    ndim = volume.ndim
    axes = list(range(ndim))
    axes.pop(source_axis)
    axes.insert(target_axis, source_axis)
    return np.transpose(volume, axes)


def _restore_frame_order(
    volume: np.ndarray, frame_order: list[int] | None, frame_axis_in: int
) -> np.ndarray:
    """Permute the frame axis to original index order.

    The cine pipeline emits frames in the order returned by the
    aggregator (typically already 0, 1, ... F-1). If the Subject records
    a different ``frame_order`` (e.g., shuffled at load), undo it here.

    .. note::

       **Dormant in production, by construction.** ``frame_order`` has a
       producer (``CineMRIDataset``, which always emits the identity list)
       and a consumer (this function), but no transport between them:
       ``pipelines/infer.py::_save_output`` is file-driven -- it calls
       ``load_tensor_from_file`` and never builds a ``tio.Subject`` -- so it
       passes no ``frame_order`` and the guard above returns immediately.
       Reaching this code needs the temporal sampler and
       ``TemporalGridAggregator`` wired into inference. Kept, not deleted,
       because that is the wiring's consumer; unit tests cover both the
       identity and shuffled paths so it stays correct until then.

       Note also that it takes an **uncollated** ``list[int]``. TorchIO's
       collate leaves the list nested (``[[0,1,2],[0,1,2]]``), which walks
       into ``inverse[original_idx]`` with a list subscript and raises
       ``TypeError``. Whoever wires the transport must pass the per-subject
       value, not the batched one.
    """
    if frame_order is None:
        return volume
    if frame_order == list(range(len(frame_order))):
        # Identity reorder — no-op.
        return volume
    # Build inverse permutation: position[i] = where frame i went.
    inverse = [0] * len(frame_order)
    for new_pos, original_idx in enumerate(frame_order):
        inverse[original_idx] = new_pos
    return np.take(volume, inverse, axis=frame_axis_in)


# ── Backend writers ──────────────────────────────────────────────────────────


def _write_nifti(path: Path, volume: np.ndarray, affine: np.ndarray | None = None) -> None:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "OutputWriter requires nibabel for NIfTI output. Install with `pip install nibabel`."
        ) from e
    affine_to_use = affine if affine is not None else np.eye(4)
    nib.save(  # type: ignore[no-untyped-call]
        nib.Nifti1Image(volume, affine=affine_to_use), str(path)
    )


def _write_h5(path: Path, volume: np.ndarray, dataset_key: str = "data") -> None:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "OutputWriter requires h5py for H5 output. Install with `pip install h5py`."
        ) from e
    with h5py.File(str(path), "w") as f:
        f.create_dataset(dataset_key, data=volume)


def _write_npy(path: Path, volume: np.ndarray) -> None:
    np.save(str(path), volume)


# ── Public API ──────────────────────────────────────────────────────────────


class OutputWriter:
    """Config-driven inference output writer (Phase 4d).

    Args:
        config: Populated :class:`ModeOutputSchema` (``data.modes.infer.output``).
        output_dir: Directory under which all writes land.

    Example::

        writer = OutputWriter(config=cfg.modes.infer.output, output_dir="./preds")
        writer.write(
            tensor=pred,                            # (C, X, Y, Z) or (F, X, Y, Z)
            subject_id="sub-001",
            file_id="sub-001_pred",
            frame_order=subject.get("frame_order"),  # None for non-cine
            affine=subject_affine,                   # optional NIfTI affine
        )
    """

    def __init__(self, config: ModeOutputSchema, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        tensor: torch.Tensor | np.ndarray,
        subject_id: str,
        file_id: str | None = None,
        frame_order: list[int] | None = None,
        affine: np.ndarray | None = None,
        idx: int = 0,
        extra_placeholders: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Write one prediction to disk per the configured format.

        Returns:
            List of paths written (one per channel for multi-channel
            outputs; one total otherwise).
        """
        volume = _to_numpy(tensor)
        if volume.ndim < 3:
            raise ValueError(f"OutputWriter expects at least 3D arrays; got shape {volume.shape}.")

        # Cine: restore frame order BEFORE channel split, since frames
        # live on the channel axis by convention.
        #
        # The two steps are independent and used to be one branch. Placing the
        # permute under ``frame_order is not None`` made ``frame_axis_out``
        # unreachable in production: the only caller of this method,
        # ``pipelines/infer.py::_save_output``, passes no ``frame_order`` at
        # all, so an arm declaring ``frame_axis_out: 3`` silently got its
        # frames written on axis 0. Axis placement needs only the config.
        if self.config.temporal.enabled:
            if frame_order is not None and self.config.temporal.preserve_frame_order:
                volume = _restore_frame_order(volume, frame_order, frame_axis_in=0)
            volume = _permute_to_frame_axis(
                volume, source_axis=0, target_axis=self.config.temporal.frame_axis_out
            )

        # Multi-channel split (Phase 4b qMRI maps).
        paths_written: list[Path] = []
        if self.config.multi_channel.enabled:
            channels = self.config.multi_channel.channels
            n_channels = volume.shape[0]
            if n_channels != len(channels):
                raise ValueError(
                    f"Multi-channel output: tensor has {n_channels} channels "
                    f"but config declares {len(channels)} "
                    f"({[c.name for c in channels]}). Channel count mismatch "
                    "is almost always a model/config drift bug."
                )
            for c_idx, channel in enumerate(channels):
                ch_volume = volume[c_idx]
                ch_transformed = _apply_channel_transform(ch_volume, channel)
                name = self._resolve_filename(
                    subject_id, file_id, channel.name, idx, extra_placeholders
                )
                path = self._write_one(name, ch_transformed, affine)
                paths_written.append(path)
        else:
            # Single-tensor write (legacy).
            name = self._resolve_filename(subject_id, file_id, "pred", idx, extra_placeholders)
            transformed = volume.astype(np.float32)
            path = self._write_one(name, transformed, affine)
            paths_written.append(path)

        return paths_written

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_filename(
        self,
        subject_id: str,
        file_id: str | None,
        name: str,
        idx: int,
        extras: dict[str, Any] | None,
    ) -> str:
        placeholders: dict[str, Any] = {
            "subject_id": subject_id,
            "file_id": file_id or subject_id,
            "name": name,
            "idx": idx,
        }
        if extras:
            placeholders.update(extras)
        return self.config.filename_template.format(**placeholders)

    def _write_one(self, basename: str, volume: np.ndarray, affine: np.ndarray | None) -> Path:
        ext = self._extension_for_format()
        path = self.output_dir / f"{basename}{ext}"
        if self.config.format == "nifti":
            _write_nifti(path, volume, affine)
        elif self.config.format == "h5":
            _write_h5(path, volume)
        elif self.config.format == "npy":
            _write_npy(path, volume)
        else:
            raise ValueError(f"Unsupported writer format: {self.config.format}")
        return path

    def _extension_for_format(self) -> str:
        return {"nifti": ".nii.gz", "h5": ".h5", "npy": ".npy"}[self.config.format]


def write_output(
    tensor: torch.Tensor | np.ndarray,
    config: ModeOutputSchema,
    output_dir: str | Path,
    subject_id: str,
    file_id: str | None = None,
    frame_order: list[int] | None = None,
    affine: np.ndarray | None = None,
    idx: int = 0,
) -> list[Path]:
    """One-shot functional wrapper around :class:`OutputWriter`.

    Convenience for callers that only need a single write — avoids
    instantiating a writer for one-off use.
    """
    writer = OutputWriter(config=config, output_dir=output_dir)
    return writer.write(
        tensor=tensor,
        subject_id=subject_id,
        file_id=file_id,
        frame_order=frame_order,
        affine=affine,
        idx=idx,
    )
