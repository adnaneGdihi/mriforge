"""The region registry: the SSOT the figure tree and leaderboard resolve against."""

from __future__ import annotations

import pytest

from mriforge.data.regions.region_registry import (
    REGION_SPECS,
    RegionTier,
    control_id_for,
    is_control_id,
    lesion_region_id,
    region_spec,
    tissue_region_ids,
)


class TestTheIdentityRegionExists:
    def test_full_is_declared(self) -> None:
        """Without a whole-slice control there is nothing for a regional ranking to
        diverge FROM, and the region axis stops being a comparison."""
        assert region_spec("full").tier is RegionTier.IDENTITY


class TestNoDefaultRegion:
    def test_an_undeclared_region_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown region"):
            region_spec("tissue:hippocampus_left")

    def test_the_error_points_at_the_ssot(self) -> None:
        with pytest.raises(KeyError, match="region_registry"):
            region_spec("nope")


class TestRectangularity:
    def test_tissue_regions_are_not_rectangular(self) -> None:
        """GM/WM are blobs, so their bbox is a strict superset. A crop-tier metric
        on one would score the bbox -- which contains WM, CSF and background -- and
        report it as 'LPIPS on grey matter'."""
        for rid in tissue_region_ids():
            assert not REGION_SPECS[rid].rectangular

    def test_the_full_slice_is_rectangular(self) -> None:
        assert region_spec("full").rectangular

    def test_pooled_lesions_are_not_assumed_rectangular(self) -> None:
        """lesion_any is a UNION of boxes, and a union of boxes need not be a box."""
        assert not region_spec("path:lesion_any").rectangular


class TestDependencies:
    def test_tissue_regions_declare_their_synthseg_dependency(self) -> None:
        """So the whole tier can be skipped wholesale when SynthSeg QC fails."""
        for rid in tissue_region_ids():
            assert REGION_SPECS[rid].requires_synthseg

    def test_pathology_regions_do_not_need_synthseg(self) -> None:
        """A SynthSeg failure must not take the pathology axis down with it."""
        assert not region_spec("path:lesion_any").requires_synthseg
        assert region_spec("path:lesion_any").requires_annotations

    def test_the_control_needs_both(self) -> None:
        """It is a lesion-sized box placed in SynthSeg-confirmed normal tissue."""
        spec = region_spec("control:lesion_any")
        assert spec.requires_annotations
        assert spec.requires_synthseg


class TestControlPairing:
    def test_the_control_id_is_derived_by_one_rule(self) -> None:
        """A mismatch here would pair a lesion with the WRONG control, and the
        paired test would still happily produce a number."""
        assert control_id_for("path:lesion_any") == "control:lesion_any"
        assert control_id_for(lesion_region_id("mass")) == "control:lesion_mass"

    def test_only_pathology_regions_get_controls(self) -> None:
        with pytest.raises(ValueError, match="not a pathology region"):
            control_id_for("tissue:gm")

    def test_control_ids_are_recognisable(self) -> None:
        assert is_control_id(control_id_for("path:lesion_any"))
        assert not is_control_id("path:lesion_any")


class TestPoolingIsTheDefault:
    def test_the_default_lesion_region_is_the_pooled_union(self) -> None:
        assert lesion_region_id() == "path:lesion_any"

    def test_per_group_regions_are_opt_in(self) -> None:
        assert lesion_region_id("hemorrhage") == "path:lesion_hemorrhage"
