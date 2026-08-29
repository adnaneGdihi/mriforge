"""RegionMask / RegionSet: geometry, and the things that must raise."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.core.metrics.regions.types import (  # noqa: E402
    RegionMask,
    RegionSet,
    RegionSource,
)


def _mask(**kw) -> RegionMask:
    m = torch.zeros(32, 32, dtype=torch.bool)
    m[4:12, 6:20] = True  # 8 x 14 rectangle
    defaults = {
        "mask": m,
        "region_id": "path:lesion#0",
        "source": RegionSource.PATHOLOGY,
        "provenance": {"annotation": "fastmri_plus"},
    }
    return RegionMask(**{**defaults, **kw})


class TestGeometry:
    def test_bbox_and_sides(self) -> None:
        r = _mask()
        assert r.bbox == (4, 6, 12, 20)
        assert r.bbox_shape == (8, 14)
        assert r.min_side == 8  # the SHORT side, not the long one
        assert r.n_px == 8 * 14

    def test_a_filled_rectangle_is_rectangular(self) -> None:
        assert _mask().is_rectangular

    def test_a_disc_is_not_rectangular(self) -> None:
        """GM/WM are blobs: their bbox is a strict superset, which is why crop-tier
        metrics are ineligible on them."""
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        disc = ((yy - 16) ** 2 + (xx - 16) ** 2) < 64
        r = _mask(mask=disc, region_id="synthseg:gm", source=RegionSource.TISSUE)
        assert not r.is_rectangular
        assert r.n_px < r.bbox_shape[0] * r.bbox_shape[1]

    def test_min_side_distinguishes_a_sliver_from_a_square(self) -> None:
        """A 1x30 sliver has decent AREA and no usable neighbourhood.

        This is why min_roi_side and min_roi_px are both needed: neither implies the
        other.
        """
        sliver = torch.zeros(32, 32, dtype=torch.bool)
        sliver[5, 1:31] = True
        r = _mask(mask=sliver)
        assert r.n_px == 30
        assert r.min_side == 1


class TestValidation:
    def test_an_empty_region_raises(self) -> None:
        """0/0 is not 0. An empty region is a data bug, not a region to score."""
        with pytest.raises(ValueError, match="region is empty"):
            _mask(mask=torch.zeros(32, 32, dtype=torch.bool))

    def test_a_float_mask_is_rejected(self) -> None:
        """A float mask invites silent partial weighting -- threshold it upstream."""
        with pytest.raises(TypeError, match="bool tensor"):
            _mask(mask=torch.ones(32, 32))

    def test_a_3d_mask_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be \[H, W\]"):
            _mask(mask=torch.ones(1, 32, 32, dtype=torch.bool))

    def test_a_non_full_region_without_provenance_raises(self) -> None:
        """An ROI you cannot reproduce is not a result."""
        with pytest.raises(ValueError, match="need provenance"):
            _mask(provenance={})

    def test_the_full_region_needs_no_provenance(self) -> None:
        assert RegionSet.full_region(16, 16).n_px == 256


class TestRegionSet:
    def test_must_contain_the_full_identity_region(self) -> None:
        """Without a whole-slice control there is nothing for a regional ranking to
        diverge FROM -- the region axis stops being a comparison."""
        with pytest.raises(ValueError, match="identity region"):
            RegionSet((_mask(),))

    def test_rejects_duplicate_region_ids(self) -> None:
        with pytest.raises(ValueError, match="duplicate region_id"):
            RegionSet((RegionSet.full_region(32, 32), _mask(), _mask()))

    def test_lookup_by_id(self) -> None:
        rs = RegionSet((RegionSet.full_region(32, 32), _mask()))
        assert rs["path:lesion#0"].n_px == 112
        assert len(rs) == 2
        with pytest.raises(KeyError, match="no region"):
            rs["nope"]
