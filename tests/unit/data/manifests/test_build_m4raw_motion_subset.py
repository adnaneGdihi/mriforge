"""Unit tests for the M4Raw motion-subset manifest builder (campaign arm C-2).

The subject-filtering logic is exercised on a tiny synthetic v3.0-style
metadata structure — no real M4Raw data is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mriforge.data.manifests.build_m4raw_motion_subset import (
    build_motion_subset_manifest,
    filter_motion_subjects,
    subject_id_from_record,
)


@pytest.fixture()
def synthetic_records() -> list[dict[str, object]]:
    """A miniature mix of clean and motion-corrupted M4Raw-style records."""
    return [
        # Clean training subject (no motion token).
        {
            "relative_path": "2022061203_T101.h5",
            "filename": "2022061203_T101.h5",
            "file_id": "2022061203_T101",
            "shape": [18, 4, 256, 256],
        },
        {
            "relative_path": "2022061203_T102.h5",
            "filename": "2022061203_T102.h5",
            "file_id": "2022061203_T102",
            "shape": [18, 4, 256, 256],
        },
        # Motion-corrupted subject under inter-scan_motion/.
        {
            "relative_path": "inter-scan_motion/2022061401_FLAIR01.h5",
            "filename": "2022061401_FLAIR01.h5",
            "file_id": "2022061401_FLAIR01",
            "shape": [18, 4, 256, 256],
        },
        {
            "relative_path": "inter-scan_motion/2022061401_T101.h5",
            "filename": "2022061401_T101.h5",
            "file_id": "2022061401_T101",
            "shape": [18, 4, 256, 256],
        },
        # Motion-corrupted subject under intra-scan_motion/.
        {
            "relative_path": "intra-scan_motion/2022061402_T201.h5",
            "filename": "2022061402_T201.h5",
            "file_id": "2022061402_T201",
            "shape": [18, 4, 256, 256],
        },
    ]


def test_subject_id_extraction() -> None:
    assert subject_id_from_record({"filename": "2022061401_FLAIR01.h5"}) == "2022061401"
    assert subject_id_from_record({"file_id": "2022061402_T201"}) == "2022061402"
    # Non-conforming names yield None rather than raising.
    assert subject_id_from_record({"filename": "not_a_subject.h5"}) is None
    assert subject_id_from_record({}) is None


def test_path_token_rule_selects_motion_records(synthetic_records) -> None:
    """With no explicit allow-list, motion is detected via path tokens."""
    kept = filter_motion_subjects(synthetic_records)
    kept_ids = {r["file_id"] for r in kept}
    assert kept_ids == {
        "2022061401_FLAIR01",
        "2022061401_T101",
        "2022061402_T201",
    }
    # The clean subject's records are excluded.
    assert not any(r["file_id"].startswith("2022061203") for r in kept)


def test_subject_allow_list_rule(synthetic_records) -> None:
    """An explicit subject set retains those subjects even without a token."""
    # Pretend the clean subject is actually a flagged motion subject.
    kept = filter_motion_subjects(
        synthetic_records,
        motion_subjects={"2022061203"},
    )
    kept_ids = {r["file_id"] for r in kept}
    # Allow-listed clean subject AND token-flagged motion subjects are kept.
    assert "2022061203_T101" in kept_ids
    assert "2022061401_FLAIR01" in kept_ids


def test_filter_does_not_mutate_input(synthetic_records) -> None:
    original_len = len(synthetic_records)
    _ = filter_motion_subjects(synthetic_records)
    assert len(synthetic_records) == original_len


def test_filter_deduplicates_by_filename() -> None:
    dup = [
        {
            "relative_path": "inter-scan_motion/2022061401_T101.h5",
            "filename": "2022061401_T101.h5",
            "file_id": "2022061401_T101",
        },
        {
            "relative_path": "inter-scan_motion/2022061401_T101.h5",
            "filename": "2022061401_T101.h5",
            "file_id": "2022061401_T101",
        },
    ]
    kept = filter_motion_subjects(dup)
    assert len(kept) == 1


def test_build_motion_subset_manifest_roundtrip(
    synthetic_records, tmp_path: Path
) -> None:
    """End-to-end: read a synthetic source manifest, write the held-out subset."""
    source = tmp_path / "m4raw_motion_src.json"
    source.write_text(
        json.dumps(
            {
                "manifest_version": "3.0",
                "dataset_name": "m4raw_motion",
                "data_root": "databases/m4raw/data/motion/motion",
                "file_type": "h5",
                "total_records": len(synthetic_records),
                "records": synthetic_records,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "held_out.json"

    manifest = build_motion_subset_manifest(source_manifest=source, output=out)

    # Returned dict and written file agree.
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["manifest_version"] == "3.0"
    assert manifest["held_out"] is True
    assert manifest["dataset_name"] == "m4raw_motion_subset_held_out"
    # Only the three motion records survive; counts are consistent.
    assert manifest["total_records"] == 3
    assert len(manifest["records"]) == 3
    assert manifest["data_root"] == "databases/m4raw/data/motion/motion"
