"""Crop tier: the tight bbox, and none of the three fabrications."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.regions.crop import crop_to_region  # noqa: E402
from spectramr.core.metrics.regions.types import RegionMask, RegionSource  # noqa: E402


def _region(mask: torch.Tensor, region_id: str = "path:lesion#0") -> RegionMask:
    return RegionMask(
        mask=mask,
        region_id=region_id,
        source=RegionSource.PATHOLOGY,
        provenance={"annotation": "test"},
    )


def test_crops_to_the_tight_bbox() -> None:
    x = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[4:12, 6:20] = True

    out = crop_to_region(x, _region(m))
    assert out.shape == (1, 1, 8, 14)
    assert torch.equal(out, x[..., 4:12, 6:20])


def test_no_resampling_the_values_are_untouched() -> None:
    """Upsampling to reach a metric's floor manufactures the high-frequency content
    the metric exists to compare. The gate declines instead; the crop never resizes."""
    g = torch.Generator().manual_seed(1)
    x = torch.rand(1, 1, 32, 32, generator=g)
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[0:5, 0:5] = True

    out = crop_to_region(x, _region(m))
    assert out.shape == (1, 1, 5, 5)  # NOT padded or upsampled to any floor
    assert torch.equal(out, x[..., 0:5, 0:5])


def test_out_of_region_pixels_inside_the_bbox_are_not_zeroed() -> None:
    """Zeroing them manufactures a hard step-edge at the region boundary, which
    LPIPS/NIQE/every gradient metric will happily 'detect' as structure."""
    x = torch.full((1, 1, 8, 8), 5.0)
    disc = torch.zeros(8, 8, dtype=torch.bool)
    disc[2:6, 2:6] = True
    disc[2, 2] = False  # a notch -> non-rectangular

    out = crop_to_region(x, _region(disc, region_id="synthseg:gm"))
    # Every value inside the bbox survives, including the notched-out corner.
    assert out.shape == (1, 1, 4, 4)
    assert torch.equal(out, torch.full((1, 1, 4, 4), 5.0))


def test_a_mask_from_a_different_slice_geometry_raises() -> None:
    """Regions are computed on the clean reference. A geometry mismatch means the
    mask belongs to another slice -- resampling it would move the region boundary."""
    x = torch.rand(1, 1, 16, 16)
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[0:4, 0:4] = True
    with pytest.raises(ValueError, match="different slice geometry"):
        crop_to_region(x, _region(m))


def test_preserves_dtype_including_complex() -> None:
    x = torch.rand(1, 1, 16, 16, dtype=torch.complex64)
    m = torch.zeros(16, 16, dtype=torch.bool)
    m[2:6, 2:6] = True
    assert crop_to_region(x, _region(m)).dtype == torch.complex64
