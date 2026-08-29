"""Tests for ``build_index_from_manifest`` (Mechanism A — spec sub-project 2).

When an experiment sets ``data.index_path``, the bart/ismrmrd/oracle loaders must
build their native index from the manifest's ``records[]`` + embedded ``data_root``
instead of re-globbing the disk — so ``index_path`` is a *read* knob for every
dataset family (no unread-knob facade, pitfall #15). This seam is the pure
path-construction half; existence is enforced downstream per-file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mriforge.data.builders.manifest_index import build_index_from_manifest


def _write_manifest(path: Path, data_root: Path, records: list[dict], file_type="bart"):
    path.write_text(
        json.dumps(
            {
                "manifest_version": "3.0",
                "dataset_name": "demo",
                "data_root": str(data_root),  # absolute → PathResolver returns as-is
                "file_type": file_type,
                "total_records": len(records),
                "records": records,
                "status": "downloaded",
            }
        )
    )


def test_bart_index_joins_data_root_and_relative_path(tmp_path: Path):
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(mf, root, [{"relative_path": "a"}, {"relative_path": "sub/b"}])

    index = build_index_from_manifest(mf, "bart")

    assert index == [str(root / "a"), str(root / "sub" / "b")]


def test_ismrmrd_index_joins_paths(tmp_path: Path):
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf, root, [{"relative_path": "scan1.mrd"}], file_type="ismrmrd"
    )

    index = build_index_from_manifest(mf, "ismrmrd")

    assert index == [str(root / "scan1.mrd")]


def test_twix_index_joins_paths(tmp_path: Path):
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf, root, [{"relative_path": "meas_MID00018.dat"}], file_type="twix"
    )

    index = build_index_from_manifest(mf, "twix")

    assert index == [str(root / "meas_MID00018.dat")]


def test_oracle_index_pairs_stack_with_shared_b0_map(tmp_path: Path):
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf, root, [{"relative_path": "s1.nii"}], file_type="oracle_bssfp"
    )

    index = build_index_from_manifest(mf, "oracle_bssfp", oracle_b0_map="maps/b0.nii")

    assert index == [{"stack": str(root / "s1.nii"), "b0_map": str(root / "maps" / "b0.nii")}]


def test_empty_records_raises(tmp_path: Path):
    """An unpopulated manifest (records[] empty) means the data was never
    extracted — fail loud, not silently empty (pitfall #9/#15)."""
    mf = tmp_path / "demo.json"
    _write_manifest(mf, tmp_path / "raw", [])
    with pytest.raises(FileNotFoundError):
        build_index_from_manifest(mf, "bart")


def test_missing_manifest_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_index_from_manifest(tmp_path / "nope.json", "bart")


def test_unknown_format_raises(tmp_path: Path):
    mf = tmp_path / "demo.json"
    _write_manifest(mf, tmp_path / "raw", [{"relative_path": "a"}])
    with pytest.raises(ValueError):
        build_index_from_manifest(mf, "weird_fmt")


def test_oracle_without_b0_map_raises(tmp_path: Path):
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf, tmp_path / "raw", [{"relative_path": "s1.nii"}], file_type="oracle_bssfp"
    )
    with pytest.raises(ValueError):
        build_index_from_manifest(mf, "oracle_bssfp")  # no oracle_b0_map


# ── F3 (smoke 2026-06-16): file_pattern filters the manifest branch ───────────


def test_file_pattern_filters_bart_records_by_stem(tmp_path: Path):
    """A mixed-acquisition manifest restricted to the matching basename stems —
    so ``data.bart.file_pattern`` is a read knob on the manifest branch too, not
    just the glob branch (pitfall #15)."""
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf,
        root,
        [
            {"relative_path": "sub/scan_05_kspace.cfl"},
            {"relative_path": "sub/scan_05_b1map.cfl"},
            {"relative_path": "sub/scan_06_b1map.cfl"},
        ],
    )

    index = build_index_from_manifest(mf, "bart", file_pattern="b1map")

    assert index == [
        str(root / "sub" / "scan_05_b1map.cfl"),
        str(root / "sub" / "scan_06_b1map.cfl"),
    ]


def test_file_pattern_none_returns_all_records(tmp_path: Path):
    """No pattern → backward-compatible (every record kept)."""
    root = tmp_path / "raw"
    mf = tmp_path / "demo.json"
    _write_manifest(
        mf, root, [{"relative_path": "a.cfl"}, {"relative_path": "b.cfl"}]
    )
    assert build_index_from_manifest(mf, "bart", file_pattern=None) == [
        str(root / "a.cfl"),
        str(root / "b.cfl"),
    ]
