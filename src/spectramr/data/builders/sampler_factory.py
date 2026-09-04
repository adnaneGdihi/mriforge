"""Phase 5 — sampler factory for TorchIO sampler variants.

Dispatches ``data.modes.<mode>.sampler.type`` to one of:

- ``"uniform"`` → :class:`torchio.data.UniformSampler` (random patches).
  This is the legacy default and what most existing YAMLs implicitly use.
- ``"label"`` → :class:`torchio.data.LabelSampler` — class-balanced
  patches from a label map. Use to oversample minority classes
  (lesions, pathology) without rebalancing the full dataset.
- ``"weighted"`` → :class:`torchio.data.WeightedSampler` — per-voxel
  probability-map driven. Use to bias sampling toward, e.g.,
  foreground voxels via a brain mask.

The factory does NOT cover ``"full"`` (no sampler — whole-volume), nor
``"grid"`` (handled by :mod:`spectramr.data.builders.inference_tiling`),
nor ``"temporal_*"`` (Phase 4c — temporal samplers live in
:mod:`spectramr.data.builders.temporal_sampler`).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Sampler types this factory knows how to build directly.
PATCH_SAMPLER_TYPES = {"uniform", "label", "weighted"}


def build_patch_sampler(sampler_config, patch_size: tuple) -> Any:
    """Construct the TorchIO sampler instance for ``sampler_config.type``.

    Args:
        sampler_config: a :class:`ModeSamplerSchema` instance carrying
            ``type``, ``samples_per_volume``, and type-specific fields.
        patch_size: spatial patch shape, threaded through to the
            TorchIO sampler constructor.

    Returns:
        A ``tio.data.PatchSampler`` subclass instance — caller wraps it
        in a ``tio.Queue`` for use in a DataLoader.

    Raises:
        ValueError: for unknown ``type`` values, or when type-specific
            required fields are missing (defensive — Pydantic validator
            should catch these earlier).
    """
    import torchio as tio

    stype = getattr(sampler_config, "type", "uniform")

    if stype not in PATCH_SAMPLER_TYPES:
        raise ValueError(
            f"sampler_factory.build_patch_sampler: type={stype!r} is not a "
            f"patch sampler kind. Valid: {sorted(PATCH_SAMPLER_TYPES)}. "
            "For 'full' (no patches), bypass the queue entirely. For "
            "'grid', call spectramr.data.builders.inference_tiling. For "
            "'temporal_*', call spectramr.data.builders.temporal_sampler "
            "(Phase 4c)."
        )

    if stype == "uniform":
        return tio.data.UniformSampler(patch_size=patch_size)

    if stype == "label":
        label_name = getattr(sampler_config, "label_name", None)
        if label_name is None:
            raise ValueError(
                "sampler.type='label' requires sampler.label_name (caught "
                "by Pydantic validator; defensive check here)."
            )
        label_probabilities = getattr(sampler_config, "label_probabilities", None)
        return tio.data.LabelSampler(
            patch_size=patch_size,
            label_name=label_name,
            label_probabilities=label_probabilities,
        )

    if stype == "weighted":
        probability_map = getattr(sampler_config, "probability_map", None)
        if probability_map is None:
            raise ValueError(
                "sampler.type='weighted' requires sampler.probability_map "
                "(caught by Pydantic validator; defensive check here)."
            )
        return tio.data.WeightedSampler(
            patch_size=patch_size,
            probability_map=probability_map,
        )

    raise AssertionError(f"unreachable: stype={stype}")


__all__ = ["PATCH_SAMPLER_TYPES", "build_patch_sampler"]
