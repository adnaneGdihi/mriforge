"""Crop-tier restriction: the tight bbox, and nothing else.

Three things this deliberately does **not** do, each of which would fabricate
signal:

* **No resampling.** Upsampling a 40x40 ROI to reach a metric's size floor
  manufactures the high-frequency content the metric is there to compare. That was
  a live bug in the sim2rank engine (it bilinearly upsampled to 161x161 and
  returned a confident MS-SSIM). The eligibility gate declines instead.
* **No padding.** Padding to a floor puts a constant field next to real tissue --
  a hard edge, plus a large uniform area that flattens any patch statistic.
* **No zeroing of out-of-region pixels inside the bbox.** For a non-rectangular
  region this is tempting and it is the worst option: it manufactures a hard
  step-edge at the region boundary, which LPIPS, NIQE and every gradient metric
  will happily "detect" as structure. It also changes with severity (the boundary
  is fixed but the tissue inside is not), so it fabricates a *severity-correlated*
  artifact.

Which is why the gate declares crop-tier metrics **ineligible** on non-rectangular
regions rather than reaching for any of these.
"""

from __future__ import annotations

import torch

from mriforge.core.metrics.regions.types import RegionMask

__all__ = ["crop_to_region"]


def crop_to_region(x: torch.Tensor, region: RegionMask) -> torch.Tensor:
    """Crop ``x`` (``[..., H, W]``) to ``region``'s tight bounding box.

    The returned tensor is a plain slice: same dtype, same values, no resample, no
    pad, no masking. Callers must have cleared the eligibility gate first -- this
    function does not check size floors, because a crop is not the place to decide
    whether the crop is meaningful.
    """
    y0, x0, y1, x1 = region.bbox
    h, w = x.shape[-2], x.shape[-1]
    if (h, w) != tuple(region.mask.shape):
        raise ValueError(
            f"{region.region_id}: region mask is {tuple(region.mask.shape)} but the "
            f"image is {(h, w)} -- a mask from a different slice geometry cannot be "
            "applied. Regions are computed on the clean reference; resample neither."
        )
    return x[..., y0:y1, x0:x1]
