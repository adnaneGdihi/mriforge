"""Tests for :mod:`spectramr.data.datasets.m4raw_slice_records` (#1757).

The pure half of the slice-level route: group records in, one record per
(group, slice) out, from header-only shape reads. Every failure shape the
expansion refuses is planted here; the loader half is exercised through the
dataset in ``test_m4raw_dataset.py``.
"""

from __future__ import annotations

import pytest

from spectramr.data.datasets.m4raw_slice_records import (
    expand_to_slice_records,
    retry_step,
    slice_count_of,
)


def _reader(shapes: dict[str, tuple[int, ...] | None]):
    """A header reader keyed by file name; ``None`` models an unreadable header."""

    def read(path):
        return shapes[str(path).rsplit("/", 1)[-1]]

    return read


def _single(patient: str, contrast: str, n_reps: int, shape) -> dict:
    return {
        "paths": [f"/d/{patient}_{contrast}{r:02d}.h5" for r in range(1, n_reps + 1)],
        "contrast": contrast,
        "patient_id": patient,
        "shape": shape,
    }


class TestExpansion:
    def test_record_count_is_the_sum_of_slice_counts(self) -> None:
        groups = [
            _single("p1", "T1", 3, (18, 4, 256, 256)),
            _single("p1", "T2", 3, (18, 4, 256, 256)),
            _single("p2", "FLAIR", 2, (12, 4, 256, 256)),
        ]
        records = expand_to_slice_records(groups, _reader({}))
        assert len(records) == 18 + 18 + 12

    def test_records_are_group_major_and_slice_ascending(self) -> None:
        groups = [_single("p1", "T1", 2, (3, 4, 8, 8)), _single("p2", "T1", 2, (2, 4, 8, 8))]
        records = expand_to_slice_records(groups, _reader({}))
        assert [(r["group_index"], r["slice_index"]) for r in records] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
        ]

    def test_each_record_keeps_the_group_and_names_its_slice(self) -> None:
        group = _single("p1", "T1", 3, (2, 4, 8, 8))
        records = expand_to_slice_records([group], _reader({}))
        for rec in records:
            assert rec["paths"] == group["paths"]
            assert rec["contrast"] == "T1" and rec["patient_id"] == "p1"
            assert rec["num_slices"] == 2
        assert [r["slice_index"] for r in records] == [0, 1]

    def test_shape_is_the_parent_shape_with_a_depth_of_one(self) -> None:
        """The queue filter's fast path compares ``shape[0]`` with the patch
        depth, so a slice record must describe the depth-1 subject it yields."""
        records = expand_to_slice_records([_single("p1", "T1", 2, (5, 4, 32, 24))], _reader({}))
        assert all(r["shape"] == (1, 4, 32, 24) for r in records)

    def test_the_group_records_are_not_mutated(self) -> None:
        group = _single("p1", "T1", 2, (2, 4, 8, 8))
        before = dict(group)
        expand_to_slice_records([group], _reader({}))
        assert group == before

    def test_an_empty_index_expands_to_nothing(self) -> None:
        assert expand_to_slice_records([], _reader({})) == []


class TestFederatedSides:
    def _pair(self, source_shape, target_shape):
        return {
            "source": ["/d/p1_T101.h5", "/d/p1_T102.h5"],
            "target": ["/d/p1_T201.h5", "/d/p1_T202.h5"],
            "patient_id": "p1",
            "target_contrast": "T2",
            "source_contrast": "T1",
            "shape": source_shape,
        }, _reader({"p1_T201.h5": target_shape})

    def test_the_target_side_is_read_from_its_own_header(self) -> None:
        record, read = self._pair((4, 2, 8, 8), (4, 2, 8, 8))
        assert slice_count_of(record, read) == (4, (4, 2, 8, 8))
        assert len(expand_to_slice_records([record], read)) == 4

    def test_sides_with_different_slice_counts_are_refused(self) -> None:
        """Planted violation: a per-slice pairing needs equal slice counts."""
        record, read = self._pair((4, 2, 8, 8), (3, 2, 8, 8))
        with pytest.raises(ValueError, match="different slice counts per side"):
            expand_to_slice_records([record], read)

    def test_an_unreadable_target_header_is_refused(self) -> None:
        record, read = self._pair((4, 2, 8, 8), None)
        with pytest.raises(ValueError, match="could not be read"):
            expand_to_slice_records([record], read)


class TestRefusals:
    def test_a_group_without_a_stamped_shape_is_refused(self) -> None:
        """Planted violation: an unreadable header means an unknown slice count,
        which is never guessed (the per-group route tolerates it; this one
        cannot build its index without it)."""
        group = _single("p1", "T1", 2, None)
        del group["shape"]
        with pytest.raises(ValueError, match="could not be read"):
            expand_to_slice_records([group], _reader({}))

    def test_a_two_dimensional_store_has_no_slice_axis(self) -> None:
        with pytest.raises(ValueError, match="no slice axis"):
            expand_to_slice_records([_single("p1", "T1", 2, (64, 64))], _reader({}))

    def test_a_record_naming_no_files_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names no files"):
            expand_to_slice_records([{"patient_id": "p1", "contrast": "T1"}], _reader({}))

    def test_every_problem_is_reported_in_one_error(self) -> None:
        """Three bad groups produce one error naming all three, not the first."""
        bad = [
            _single("p1", "T1", 2, (64, 64)),
            _single("p2", "T2", 2, (64, 64)),
            _single("p3", "FLAIR", 2, (64, 64)),
        ]
        with pytest.raises(ValueError, match="3 of 3 group") as exc:
            expand_to_slice_records(bad, _reader({}))
        for label in ("p1/T1", "p2/T2", "p3/FLAIR"):
            assert label in str(exc.value)


class TestRetryStep:
    def test_a_group_record_steps_by_one(self) -> None:
        assert retry_step({"paths": ["/d/a.h5"]}) == 1

    def test_a_slice_record_steps_over_its_group(self) -> None:
        assert retry_step({"num_slices": 18, "slice_index": 4}) == 18

    def test_a_degenerate_count_still_advances(self) -> None:
        assert retry_step({"num_slices": 0}) == 1
