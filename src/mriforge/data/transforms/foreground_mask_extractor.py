"""TorchIO transform that attaches a brain foreground mask to a Subject.

Two sources, in order of preference:

1. **Sidecar mask file**: a precomputed brain mask living next to
   the input volume with one of the conventional suffixes
   (``_brain_mask.nii.gz`` per
   :mod:`mriforge.shared.utils.preprocessing_qc`, ``_mask.nii.gz``, or
   ``_brainmask.nii.gz``). This is the canonical pre-processed
   path used by the rest of the codebase.
2. **Lazy extraction**: if no sidecar file is found and
   ``allow_lazy=True``, call
   :class:`mriforge.domain.services.brain_extraction_service.BrainExtractionService`
   on the in-memory tensor and cache the result alongside the
   source file (when the file path is available).

The mask becomes available downstream as
``subject[mask_name]`` (a :class:`tio.LabelMap`), so strategies
and losses can apply it without re-running brain extraction every
iteration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torchio as tio

from mriforge.data.transforms.registry import register_transform

logger = logging.getLogger(__name__)


_SIDECAR_SUFFIXES: tuple[str, ...] = (
    "_brain_mask.nii.gz",
    "_brain_mask.nii",
    "_brainmask.nii.gz",
    "_brainmask.nii",
    "_mask.nii.gz",
    "_mask.nii",
)


def _strip_nifti_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _find_sidecar_mask(volume_path: Path) -> Path | None:
    stem = _strip_nifti_suffix(volume_path)
    parent = volume_path.parent
    for suffix in _SIDECAR_SUFFIXES:
        candidate = parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


@register_transform("foreground_mask", produces=("foreground_mask",))
class ForegroundMaskExtractor(tio.Transform):
    """Attach a brain foreground mask to a TorchIO Subject.

    Args:
        source_image: name of the image in ``subject`` to derive
            the mask from. Defaults to ``"input"``.
        mask_name: name to attach the mask under. Defaults to
            ``"foreground_mask"``.
        threshold: intensity threshold used by the lazy
            simple-thresholding fallback (when neither a sidecar
            file nor a brain-extraction service is available).
            Pixels with ``volume > threshold * percentile_99``
            count as foreground.
        allow_lazy: if True, fall back to in-memory extraction
            when no sidecar mask is found.
        include / exclude / copy / kwargs: standard
            :class:`tio.Transform` arguments.
    """

    def __init__(
        self,
        source_image: str = "input",
        mask_name: str = "foreground_mask",
        threshold: float = 0.05,
        allow_lazy: bool = True,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(include=include, exclude=exclude, copy=copy, **kwargs)
        self.source_image = source_image
        self.mask_name = mask_name
        self.threshold = float(threshold)
        self.allow_lazy = allow_lazy

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.source_image not in subject:
            logger.warning(
                "[ForegroundMaskExtractor] source image '%s' missing from subject; "
                "skipping mask attachment.",
                self.source_image,
            )
            return subject
        if self.mask_name in subject:
            # Already attached by an upstream transform — leave as-is.
            return subject

        image = subject[self.source_image]
        # 1) Sidecar file lookup (only possible if the source image
        #    has a backing path on disk).
        path = getattr(image, "path", None)
        if path is not None:
            sidecar = _find_sidecar_mask(Path(path))
            if sidecar is not None:
                subject.add_image(tio.LabelMap(str(sidecar), affine=image.affine), self.mask_name)
                return subject

        # 2) Lazy in-memory threshold fallback. Cheap, deterministic,
        #    no external service. The strategy can swap in a
        #    BrainExtractionService-backed mask before training.
        if self.allow_lazy:
            data = image.data
            mask_tensor = self._threshold_mask(data, self.threshold)
            subject.add_image(
                tio.LabelMap(tensor=mask_tensor, affine=image.affine),
                self.mask_name,
            )
            return subject

        logger.warning(
            "[ForegroundMaskExtractor] no sidecar mask for '%s' and lazy fallback "
            "is disabled; mask not attached.",
            getattr(image, "path", "<in-memory>"),
        )
        return subject

    @staticmethod
    def _threshold_mask(data: torch.Tensor, threshold: float) -> torch.Tensor:
        """Robust intensity-threshold mask (foreground = large signal)."""
        # data is (C, W, H, D) per TorchIO convention. Use the magnitude
        # of the first channel as the proxy.
        intensity = data[0].abs().float()
        # Compute the 99-percentile over non-zero voxels only. Background
        # in MRI volumes can dominate the histogram (especially in
        # skull-stripped or sparsely-sampled data), which would collapse
        # the percentile to ~0 and let noise voxels pass the threshold.
        nonzero = intensity[intensity > 0]
        if nonzero.numel() == 0:
            return torch.zeros_like(intensity, dtype=data.dtype).unsqueeze(0)
        upper = torch.quantile(nonzero.flatten(), 0.99).clamp_min(1e-6)
        mask = (intensity > threshold * upper).to(data.dtype)
        return mask.unsqueeze(0)


__all__ = ["ForegroundMaskExtractor"]
