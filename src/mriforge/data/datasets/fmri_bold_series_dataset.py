"""BOLD time-series fMRI dataset.

Split out of :mod:`fmri_dataset` in the Wave 0 exit-criterion work (#1400).
Reachable under its original spelling -- ``fmri_dataset`` re-exports it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mriforge.data.datasets.fmri_volume_dataset import _read_volume

logger = logging.getLogger(__name__)


def _sibling_target(volume: Path, suffix: str) -> Path | None:
    """The companion target beside ``volume``, or None if absent.

    ``sub-01_bold.nii.gz`` + ``_target`` -> ``sub-01_bold_target.nii.gz``.
    Handles the double extension NIfTI uses.
    """
    name = volume.name
    for ext in (".nii.gz", ".nii", ".npy"):
        if name.endswith(ext):
            candidate = volume.with_name(f"{name[: -len(ext)]}{suffix}{ext}")
            return candidate if candidate.exists() else None
    return None


class FMRIBoldSeriesDataset(Dataset):
    """4-D BOLD series -> ``tio.Subject``, with the time axis kept legible.

    The Subject-emitting sibling of :class:`FMRIVolumeDataset`, which returns a
    plain dict and therefore cannot travel the ``DataPipelineDirector`` route
    (every registered factory threads a ``tio.Compose`` through and yields
    Subjects). That mismatch is why no ``dataset_type`` selected the temporal
    loader, and why ``mri_functional`` / ``mri_dynamic`` / ``mri_perfusion``
    were declarable on zero arms in the corpus (issue #998).

    **Why the time axis survives here and not through ``dataset_type: nifti``.**
    TorchIO's data model is ``[C, H, W, D]``, so a ``[T, H, W, D]`` series has to
    put ``T`` in the channel slot either way -- that is not the difference.
    ``NiftiStrategy.load`` folds the trailing axis into channels *and drops
    everything that made it a time axis*: no TR, no frame order, no count. What
    survives is indistinguishable from coils or contrasts, which is exactly why
    ``axis_exposure`` annotates ``nifti`` as exposing nothing.

    This dataset instead preserves the semantics alongside the tensor --
    ``frame_order``, ``num_frames`` and ``tr`` ride on the Subject -- so a
    consumer can tell that channel *i* is time *i*. That is the same standard
    the ``cine`` row already meets ("preserves frame ordering as
    ``subject['frame_order']``, so the frame axis is real and usable"), and it
    is what makes ``DATASET_TYPE_AXES['fmri'] = {TEMPORAL}`` a true claim rather
    than the wrong annotation the module's contract warns against.
    """

    def __init__(
        self,
        index: list[dict[str, str]],
        transform: Callable[[Any], Any] | None = None,
        *,
        tr_seconds: float = 0.72,
        phase_encode_axis: int = -2,
        target_source: str | None = None,
        target_suffix: str = "_target",
    ) -> None:
        super().__init__()
        self.index = list(index)
        self.transform = transform
        self.tr_seconds = float(tr_seconds)
        self.phase_encode_axis = int(phase_encode_axis)
        # The pairing is DECLARED or the arm does not run. An earlier revision of
        # this class emitted ``target = input.clone()`` -- the "degradation twin"
        # pattern FieldRefDataset uses legitimately, because a degradation
        # transform sits downstream of it. Nothing degrades a BOLD series here,
        # so that twin is the degenerate case both fMRI strategies solve
        # trivially: BeltramiEPIDistortion is minimised analytically at
        # Delta_B0 = 0 (residual and mu_reg vanish together) and
        # SpatiotemporalAdaptiveSFCRecon becomes the identity. Loss falls
        # smoothly, metrics look excellent, the field map is worthless -- pitfall
        # #16 at the moment of wiring. See
        # TODO/inprogress/backlog_fmri_serving_path_2026_08_05.md #2.
        if target_source != "sibling":
            raise ValueError(
                "FMRIBoldSeriesDataset requires data.fmri.target_source: sibling "
                "-- a paired acquisition (blip-up/blip-down is the standard fMRI "
                f"answer), matched by data.fmri.target_suffix. Got "
                f"{target_source!r}. There is no self-pairing option: with "
                "target == input both fMRI strategies have a trivial minimum and "
                "would train to a worthless answer while every metric improved."
            )
        self.target_source = target_source
        self.target_suffix = str(target_suffix)

    def __len__(self) -> int:
        return len(self.index)

    def dry_iter(self) -> list[Any]:
        """Cheap Subject shells for ``tio.Queue`` length probing (no voxel read).

        Mirrors ``FieldRefDataset.dry_iter`` / ``OracleBssfpDataset.dry_iter``;
        without it a queued arm dies at step 0 on a missing attribute.
        """
        import torchio as tio

        stub = torch.zeros(1, 1, 1, 1)
        return [tio.Subject(input=tio.ScalarImage(tensor=stub)) for _ in range(len(self))]

    def __getitem__(self, idx: int) -> Any:
        import torchio as tio

        path = Path(self.index[idx]["volume"])
        arr = _read_volume(path)
        if arr is None:
            # Same refusal as FMRIVolumeDataset: a zeros placeholder trains the
            # model on fabricated data while the run looks healthy (#9/#16).
            raise RuntimeError(
                f"FMRIBoldSeriesDataset: failed to load volume {str(path)!r}. "
                "Refusing to substitute a zeros placeholder."
            )
        if arr.ndim != 4:
            # The whole point of this route is the trailing time axis. A 3-D
            # volume here is a misrouted arm, and accepting it would make the
            # TEMPORAL annotation false for that sample (pitfall #9).
            raise ValueError(
                f"FMRIBoldSeriesDataset expects a 4-D BOLD series, got shape "
                f"{arr.shape} from {str(path)!r}. Use dataset_type='nifti' for "
                "3-D volumes -- it does not claim a temporal axis."
            )
        # NIfTI stores (H, W, D, T); TorchIO needs the non-spatial axis first.
        x = torch.from_numpy(arr).float().permute(3, 0, 1, 2).contiguous()
        num_frames = int(x.shape[0])
        target_path = _sibling_target(path, self.target_suffix)
        if target_path is None:
            raise FileNotFoundError(
                f"FMRIBoldSeriesDataset: no sibling target for {str(path)!r} "
                f"(expected suffix {self.target_suffix!r}). Refusing to fall back "
                "to target = input, which both fMRI strategies minimise trivially."
            )
        target_arr = _read_volume(target_path)
        if target_arr is None or target_arr.ndim != 4:
            raise ValueError(
                f"FMRIBoldSeriesDataset: sibling target {str(target_path)!r} is "
                f"unreadable or not 4-D (got "
                f"{None if target_arr is None else target_arr.shape})."
            )
        y = torch.from_numpy(target_arr).float().permute(3, 0, 1, 2).contiguous()
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=x),
            target=tio.ScalarImage(tensor=y),
            # The semantics that make the channel slot readable as time.
            frame_order=list(range(num_frames)),
            num_frames=num_frames,
            tr=self.tr_seconds,
            phase_encode_axis=self.phase_encode_axis,
            source_path=str(path),
        )
        if self.transform is not None:
            subject = self.transform(subject)
        return subject


__all__ = ["FMRIBoldSeriesDataset"]
