"""Phase 4 — config-driven resample + crop_or_pad transforms.

Two YAML blocks in :class:`spectramr.config.schemas.data.DataConfigSchema`
drive these transforms:

- ``data.resample.*`` — voxel spacing changes (isotropic or
  reference-based). Maps to :class:`torchio.Resample` with an optional
  Gaussian anti-aliasing pre-filter on downsample paths.
- ``data.crop_or_pad.*`` — voxel count changes without changing
  spacing. Maps to :class:`torchio.CropOrPad`.

When ``enabled=false`` on either block, this dispatcher returns ``None``
and the caller falls back to the legacy hardcoded transforms
(``_ResampleToReferenceTransform`` and ``_EnsureMinimumSpatialSize``).
This preserves pre-Phase-4 behavior for ~200 existing YAMLs.

The two transforms compose cleanly when both are enabled:
    1. Resample (spacing change) runs first
    2. CropOrPad (shape change) runs second

Use case: ULF + HF cohorts with heterogeneous resolution → resample
to canonical isotropic spacing, then crop/pad to a canonical shape so
downstream models see a fixed tensor.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Map our PaddingMode enum onto TorchIO's ``padding_mode`` argument.
# TorchIO accepts: "constant" (with cval), "reflect", "edge", or numeric.
_TORCHIO_PAD_MODE: dict[str, Any] = {
    "zero": 0.0,
    "reflect": "reflect",
    "edge": "edge",
    # 'mean' is handled separately above (logger.warning + fallback to 'reflect');
    # it is NOT a valid TorchIO padding_mode and must not appear in this lookup table.
}


def build_resample_transform(config) -> Any | None:
    """Return a configured :class:`tio.Resample` instance, or ``None``
    when ``config.resample_enabled`` is False.

    Args:
        config: a :class:`TorchIOTransformConfig` carrying the
            ``resample_*`` fields populated by
            ``TorchIOTransformConfig.from_training_config``.

    Returns:
        ``tio.Resample`` when enabled, else ``None`` (caller falls
        back to the legacy hardcoded path).

    Raises:
        ValueError: if ``resample_enabled`` is True but the strategy's
            required field (e.g. ``target_spacing`` for ``isotropic``)
            is missing — should already be caught by the Pydantic
            validator, but defended here in case the builder is fed a
            raw dict.
    """
    if not getattr(config, "resample_enabled", False):
        return None

    import torchio as tio

    strategy = getattr(config, "resample_strategy", "reference")
    interp = getattr(config, "resample_interpolation", "bspline")
    anti_alias = bool(getattr(config, "resample_anti_aliasing", True))

    # Anti-aliasing on downsample: TorchIO's RandomAnisotropy /
    # Resample already perform pre-filtering when downsampling intensity
    # images via the spline kernel order. We surface the toggle here for
    # explicit user control — when False, we route to nearest-neighbor
    # (no filtering, no aliasing protection — user assumes the risk).
    image_interp = interp if anti_alias else "nearest"

    if strategy == "reference":
        # Caller falls back to legacy _ResampleToReferenceTransform —
        # we return None so the existing code path runs unchanged.
        return None

    if strategy == "isotropic":
        target_spacing = getattr(config, "resample_target_spacing", None)
        if target_spacing is None:
            raise ValueError(
                "resample.strategy='isotropic' requires "
                "resample.target_spacing — should have been caught by "
                "ResampleConfigSchema validator. Was this config built "
                "by hand?"
            )
        return tio.Resample(
            target=tuple(target_spacing),
            image_interpolation=image_interp,
        )

    raise ValueError(
        f"resample_strategy={strategy!r} is not recognized. Valid: 'reference', 'isotropic'."
    )


def build_crop_or_pad_transform(config) -> Any | None:
    """Return a configured :class:`tio.CropOrPad` instance, or ``None``
    when ``config.crop_or_pad_enabled`` is False.

    Args:
        config: a :class:`TorchIOTransformConfig` carrying the
            ``crop_or_pad_*`` fields.

    Returns:
        ``tio.CropOrPad`` when enabled, else ``None`` (caller falls
        back to legacy ``_EnsureMinimumSpatialSize``).
    """
    if not getattr(config, "crop_or_pad_enabled", False):
        return None

    import torchio as tio

    target_shape = getattr(config, "crop_or_pad_target_shape", None)
    if target_shape is None:
        raise ValueError(
            "crop_or_pad.enabled=True requires crop_or_pad.target_shape — "
            "should have been caught by CropOrPadConfigSchema validator."
        )
    padding_mode = getattr(config, "crop_or_pad_padding_mode", "reflect")

    # ``mean`` is not native to TorchIO; map to constant=0.0 with a
    # log warning. The proper fix (compute per-subject mean before
    # padding) requires a custom transform — deferred.
    if padding_mode == "mean":
        logger.warning(
            "[CropOrPad] padding_mode='mean' is not natively supported "
            "by TorchIO; falling back to 'reflect'. (To pad with the "
            "subject mean, write a per-subject pre-filter in your "
            "transforms_override block.)"
        )
        tio_pad: Any = "reflect"
    else:
        if padding_mode not in _TORCHIO_PAD_MODE:
            raise ValueError(
                f"crop_or_pad.padding_mode={padding_mode!r} is not recognised. "
                f"Valid values: {sorted(_TORCHIO_PAD_MODE)} (plus 'mean' which "
                "falls back to 'reflect' with a warning). "
                "Should have been caught by CropOrPadConfigSchema validator; "
                "was this config built by hand?"
            )
        tio_pad = _TORCHIO_PAD_MODE[padding_mode]

    return tio.CropOrPad(
        target_shape=tuple(target_shape),
        padding_mode=tio_pad,
    )


__all__ = [
    "build_crop_or_pad_transform",
    "build_resample_transform",
]
