"""fastMRI+ parser: the y-origin flip, and the promise that no row vanishes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mriforge.data.annotations.fastmri_plus import (  # noqa: E402
    ESTABLISHED_Y_ORIGIN,
    BoxOutOfBoundsError,
    LesionBox,
    UnmappedLabelsError,
    YOrigin,
    group_boxes_by_slice,
    parse_annotations,
    pooled_lesion_mask,
)
from mriforge.data.annotations.fastmri_plus_classes import LesionGroup  # noqa: E402

_HEADER = "file,slice,study_level,x,y,width,height,label\n"


def _csv(tmp_path, rows: str, name: str = "brain.csv"):
    path = tmp_path / name
    path.write_text(_HEADER + rows)
    return path


def _box(**kw) -> LesionBox:
    defaults = {
        "file_id": "file_brain_001",
        "slice_index": 4,
        "x": 10,
        "y_raw": 20,
        "width": 30,
        "height": 40,
        "label": "Edema",
        "group": LesionGroup.EDEMA,
        "y_origin": YOrigin.TOP_LEFT,
    }
    return LesionBox(**{**defaults, **kw})


class TestYOriginIsTheOnlyConversionSite:
    """The footgun. A flipped box lands in mirror-image normal tissue, and the
    sweep then reports a confident null result that means nothing."""

    def test_top_left_is_the_identity(self) -> None:
        assert _box(y_raw=20, height=40).top_left_y(320) == 20

    def test_bottom_left_flips_about_the_image_height(self) -> None:
        # Box spans rows [20, 60) counted UP from the bottom of a 320-row image,
        # so its top row in array coordinates is 320 - 20 - 40 = 260.
        b = _box(y_raw=20, height=40, y_origin=YOrigin.BOTTOM_LEFT)
        assert b.top_left_y(320) == 260

    def test_the_two_conventions_disagree_and_that_is_the_whole_point(self) -> None:
        top = _box(y_origin=YOrigin.TOP_LEFT)
        bottom = _box(y_origin=YOrigin.BOTTOM_LEFT)
        assert top.top_left_y(320) != bottom.top_left_y(320)

    def test_the_flip_is_an_involution(self) -> None:
        """Flipping a bottom-left box's raw y through H and re-reading it as
        top-left must give the same rows -- a sanity check on the arithmetic."""
        h, height, y_raw = 320, 40, 20
        b = _box(y_raw=y_raw, height=height, y_origin=YOrigin.BOTTOM_LEFT)
        equivalent = _box(y_raw=b.top_left_y(h), height=height)
        assert equivalent.top_left_y(h) == b.top_left_y(h)

    def test_the_established_fastmri_plus_convention_is_bottom_left(self) -> None:
        """fastMRI+ boxes index a vertically FLIPPED copy of reconstruction_rss.

        Cited, not asserted: upstream's example notebook applies ``img[:, ::-1, :]``
        ("flipped up down") before drawing any box, and the README states the arrays
        were flipped up/down during DICOM conversion. We never flip the image, so the
        box flips instead. Corrected 2026-07-13 (was TOP_LEFT, on prose alone).
        """
        assert ESTABLISHED_Y_ORIGIN is YOrigin.BOTTOM_LEFT

    def test_the_established_origin_actually_flips(self) -> None:
        b = _box(y_raw=20, height=40, y_origin=ESTABLISHED_Y_ORIGIN)
        assert b.top_left_y(320) == 320 - 20 - 40
        assert (
            b.top_left_y(320) != b.y_raw
        ), "the established origin must NOT be the identity"

    def test_parse_annotations_still_requires_an_explicit_y_origin(self) -> None:
        """Kept required even though the answer is known.

        Not because it is in doubt, but so the value is always explicit and always
        stamped into the manifest's provenance (pitfall #15). A parser-level default
        would let the convention go invisible again the moment someone points this at
        a different annotation export -- which is how it got lost the first time.
        """
        with pytest.raises(TypeError):
            parse_annotations("x.csv")  # type: ignore[call-arg]


class TestTheBoxLandsOnTheFindingItAnnotates:
    """The regression that the old ``assert CONST is CONST`` could never catch.

    An asymmetric phantom: one bright blob, off-centre in the row axis. We simulate what
    a fastMRI+ annotator did -- they drew the box on ``img[::-1]`` -- and then require
    that our parser puts the mask back on the blob in the *unflipped* array we score.

    Under the wrong origin the mask lands on background, fully in bounds, raising
    nothing. That is the entire failure mode, reproduced in eight lines.
    """

    H = W = 64
    BLOB_ROWS = slice(10, 20)  # deliberately NOT row-symmetric about H/2
    BLOB_COLS = slice(30, 40)

    def _phantom(self):
        img = torch.zeros(self.H, self.W)
        img[self.BLOB_ROWS, self.BLOB_COLS] = 1.0
        return img

    def _box_as_the_annotator_drew_it(self, y_origin: YOrigin) -> LesionBox:
        """The annotator saw ``img[::-1]``, so the blob appeared at flipped rows
        ``[H-20, H-10)`` = ``[44, 54)`` and they wrote ``y=44, height=10``."""
        return _box(
            x=30,
            y_raw=self.H - self.BLOB_ROWS.stop,  # 44
            width=10,
            height=self.BLOB_ROWS.stop - self.BLOB_ROWS.start,  # 10
            y_origin=y_origin,
        )

    def test_the_established_origin_puts_the_mask_on_the_blob(self) -> None:
        img = self._phantom()
        mask = self._box_as_the_annotator_drew_it(ESTABLISHED_Y_ORIGIN).to_bool_mask(
            self.H, self.W
        )
        assert (
            float(img[mask].mean()) == 1.0
        ), "the box must cover the finding, not miss it"
        assert int(mask.sum()) == int((img > 0).sum()), "and cover it exactly"

    def test_the_wrong_origin_puts_the_mask_on_pure_background(self) -> None:
        img = self._phantom()
        mask = self._box_as_the_annotator_drew_it(YOrigin.TOP_LEFT).to_bool_mask(
            self.H, self.W
        )
        assert float(img[mask].sum()) == 0.0, "the flipped box lands on background"

    def test_and_the_wrong_origin_stays_in_bounds_so_nothing_raises(self) -> None:
        """Why no existing guard fires. A mirrored box on a brain slice is still a
        plausible box on brain -- BoxOutOfBoundsError only catches a box that misses
        the image entirely, which a mirrored one does not."""
        box = self._box_as_the_annotator_drew_it(YOrigin.TOP_LEFT)
        assert not box.is_clipped(self.H, self.W)
        box.to_bool_mask(self.H, self.W)  # does not raise -- that is the whole problem


class TestRasterisation:
    def test_mask_covers_exactly_the_box(self) -> None:
        m = _box(x=10, y_raw=20, width=30, height=40).to_bool_mask(320, 320)
        assert m.dtype is torch.bool
        assert int(m.sum()) == 30 * 40
        assert bool(m[20:60, 10:40].all())

    def test_a_partially_outside_box_is_clipped_not_dropped(self) -> None:
        b = _box(x=310, y_raw=0, width=30, height=10)
        assert b.is_clipped(320, 320)
        assert int(b.to_bool_mask(320, 320).sum()) == 10 * 10  # 320-310 = 10 cols

    def test_a_fully_outside_box_raises(self) -> None:
        """The loudest available signal that the y-origin is wrong."""
        b = _box(x=10, y_raw=400, width=10, height=10)
        with pytest.raises(BoxOutOfBoundsError, match="does not intersect"):
            b.to_bool_mask(320, 320)

    def test_a_zero_area_box_raises(self) -> None:
        with pytest.raises(ValueError, match="non-positive extent"):
            _box(width=0)

    def test_pooled_mask_is_the_union(self) -> None:
        a = _box(x=0, y_raw=0, width=10, height=10)
        b = _box(x=5, y_raw=5, width=10, height=10)  # overlaps a
        pooled = pooled_lesion_mask([a, b], 320, 320)
        assert int(pooled.sum()) == 100 + 100 - 25  # inclusion-exclusion


class TestStudyLevelRowsAreDetectedByTheColumn:
    """The shipped CSV leaves study-level geometry EMPTY, not -1.

    The parser looked only for a ``-1`` sentinel, which does not exist in the real file.
    So all 643 study-level rows failed ``int("")``, landed in ``malformed``, and the
    report -- whose entire promise is that no row vanishes uncounted -- filed them under
    the wrong heading. The ``study_level`` column sat in the required-columns list and was
    never read. Fixed 2026-07-13.
    """

    def test_an_empty_geometry_study_level_row_is_recognised(self, tmp_path) -> None:
        """Exactly the shipped format:
        ``file_brain_AXFLAIR_201_6002914,0,Yes,,,,,Global label: ...``"""
        path = _csv(
            tmp_path, "f1,0,Yes,,,,,Global label: Small vessel ischemic change\n"
        )
        r = parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN)
        assert r.study_level_dropped == 1
        assert (
            len(r.malformed) == 0
        ), "an empty-geometry study-level row is not malformed"
        assert r.n_boxes == 0

    def test_a_study_level_label_never_needs_a_declared_group(self, tmp_path) -> None:
        """The row is dropped before ``group_for`` sees it. 'Global label: ...' is not a
        LesionGroup and must never have to be one."""
        path = _csv(tmp_path, "f1,0,Yes,,,,,Global label: Whatever\n")
        parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN).raise_if_unmapped()

    def test_the_minus_one_sentinel_survives_as_a_secondary_guard(
        self, tmp_path
    ) -> None:
        """Some re-exports do use -1. Without this, a -1 box rasterises as a real ROI at a
        negative offset."""
        path = _csv(tmp_path, "f1,3,No,-1,-1,-1,-1,Edema\n")
        r = parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN)
        assert r.study_level_dropped == 1
        assert r.n_boxes == 0


class TestNoRowVanishes:
    """The conservation law. A parser that silently skips rows makes the lesion
    count unfalsifiable -- 'this cohort has 40 lesions' becomes indistinguishable
    from 'it has 300 and the parser ate 260'."""

    def test_every_row_is_accounted_for(self, tmp_path) -> None:
        path = _csv(
            tmp_path,
            "f1,3,No,10,20,30,40,Edema\n"  # kept
            "f1,3,Yes,,,,,Global label: Something\n"  # study-level (EMPTY geom, as shipped)
            "f2,5,No,1,2,3,4,Nonspecific white matter lesion\n"  # kept
            "f2,6,No,x,2,3,4,Edema\n"  # malformed geometry
            "f2,7,No,1,2,3,4,Wormholes\n"  # unmapped label
            "f2,9,No,208,83,0,0,Posttreatment change\n"  # zero-area -- 2 REAL ones in the CSV
            "f2,8,No,1,2,3,4,\n",  # empty label -> malformed
        )
        r = parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN)

        assert r.total_rows == 7
        accounted = (
            r.n_boxes
            + r.study_level_dropped
            + len(r.malformed)
            + sum(r.unmapped_labels.values())
        )
        assert accounted == r.total_rows, "a row vanished without being counted"
        assert r.n_boxes == 2
        assert r.study_level_dropped == 1
        assert len(r.malformed) == 3  # bad geometry, zero-area box, empty label
        assert r.unmapped_labels == {"Wormholes": 1}

    def test_a_zero_area_box_is_counted_not_silently_dropped(self, tmp_path) -> None:
        """The shipped CSV really carries two of these -- a 0x0 and a 1x0 box. They are a
        data bug, not a region to score, and they are COUNTED as malformed."""
        path = _csv(tmp_path, "f1,10,No,208,83,0,0,Posttreatment change\n")
        r = parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN)
        assert r.n_boxes == 0
        assert len(r.malformed) == 1
        assert "non-positive extent" in r.malformed[0][1]

    def test_study_level_rows_are_not_expanded_into_whole_slice_boxes(
        self, tmp_path
    ) -> None:
        """A whole-slice 'lesion' box is just the brain region under another name --
        it would answer a different question while looking like this one."""
        path = _csv(tmp_path, "f1,3,Yes,,,,,Edema\n")
        r = parse_annotations(path, y_origin=ESTABLISHED_Y_ORIGIN)
        assert r.n_boxes == 0
        assert r.study_level_dropped == 1

    def test_a_missing_column_refuses_to_guess_the_layout(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("file,slice,x,y\nf1,1,2,3\n")
        with pytest.raises(ValueError, match="missing expected fastMRI\\+ column"):
            parse_annotations(path, y_origin=YOrigin.TOP_LEFT)


class TestTheBuildGate:
    def test_unmapped_labels_do_not_raise_during_parsing(self, tmp_path) -> None:
        """Collection is total -- otherwise --report-labels could not exist and you
        would be back to guessing label strings."""
        path = _csv(tmp_path, "f1,3,No,1,2,3,4,Wormholes\n")
        r = parse_annotations(path, y_origin=YOrigin.TOP_LEFT)
        assert r.unmapped_labels == {"Wormholes": 1}

    def test_but_the_gate_raises_before_a_manifest_is_built(self, tmp_path) -> None:
        path = _csv(tmp_path, "f1,3,No,1,2,3,4,Wormholes\n")
        r = parse_annotations(path, y_origin=YOrigin.TOP_LEFT)
        with pytest.raises(UnmappedLabelsError, match="Wormholes"):
            r.raise_if_unmapped()

    def test_a_clean_parse_passes_the_gate(self, tmp_path) -> None:
        path = _csv(tmp_path, "f1,3,No,1,2,3,4,Edema\n")
        parse_annotations(path, y_origin=YOrigin.TOP_LEFT).raise_if_unmapped()


class TestGrouping:
    def test_boxes_group_by_file_and_slice(self, tmp_path) -> None:
        path = _csv(
            tmp_path,
            "f1,3,No,1,2,3,4,Edema\nf1,3,No,9,9,3,4,Mass\nf1,4,No,1,2,3,4,Edema\n",
        )
        r = parse_annotations(path, y_origin=YOrigin.TOP_LEFT)
        by_slice = group_boxes_by_slice(r.boxes)
        assert len(by_slice[("f1", 3)]) == 2
        assert len(by_slice[("f1", 4)]) == 1
        assert r.n_annotated_slices == 2
        assert r.n_files == 1

    def test_provenance_dict_records_every_drop(self, tmp_path) -> None:
        path = _csv(tmp_path, "f1,3,No,1,2,3,4,Edema\nf1,3,Yes,-1,-1,-1,-1,Mass\n")
        d = parse_annotations(path, y_origin=YOrigin.TOP_LEFT).to_dict()
        assert d["y_origin"] == "top_left"  # the knob is stamped (pitfall #15)
        assert d["study_level_dropped"] == 1
        assert d["boxes_parsed"] == 1
        assert d["by_group"] == {"edema": 1}
