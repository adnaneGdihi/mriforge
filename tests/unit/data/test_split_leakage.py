"""Unit tests for the train/val split-leakage analyzer (data SSOT).

``analyze_split_leakage`` is the data-layer half of the audit's
``check_train_val_split_leakage`` guard. It must:

* Detect the real, statically-detectable leak — the SAME subject (or the
  same on-disk file) present in BOTH the train manifest and the val
  manifest (the two-independent-files ``manifest`` split, which no code
  guarantees disjoint — the 2026-07-07 mrixfields cluster failure family).
* Support EVERY split strategy without crashing or false-positiving:
  ``random`` (single-index, deterministic ``split_index`` — only a subject
  straddling the boundary leaks), ``loso`` (a subject appearing at both the
  holdout site and elsewhere), ``directory``/``auto`` (folder-disjoint by
  construction — skip with info).
* Skip (never fail, never falsely pass) when the manifest files are absent
  locally — they are gitignored and regenerated on the cluster, so the
  check runs for real there.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tests.utils.data_config_stub import DataConfigStub
from typing import Any

from mriforge.data.split_leakage import (
    SplitLeakageReport,
    analyze_split_leakage,
    read_manifest_records,
)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"data_root": ".", "records": records}))


def _rec(subject_id: str, *, field: float = 3.0, contrast: str = "T1w") -> dict[str, Any]:
    rel = f"{subject_id}/{contrast}/{field}/vol.nii.gz"
    return {
        "relative_path": rel,
        "file_id": rel,
        "field_strength": field,
        "contrast": contrast,
        "subject_id": subject_id,
        "pairing_group": f"{subject_id}|{contrast}",
    }


def _cfg(
    *,
    index_path: str | None = None,
    validation_index_path: str | None = None,
    validation_split: float = 0.1,
    holdout_site: str | None = None,
    split_strategy: str = "manifest",
) -> Any:
    # The SHARED stub, not a hand-rolled namespace. This one had been migrated
    # for `data.split.*` but not for `data.source.*`, so every reader of
    # `data.source.index_path` hit "'SimpleNamespace' object has no attribute
    # 'source'". DataConfigStub carries every sub-block and routes the flat
    # legacy kwargs from RENAMES, so it cannot fall behind the next fold the way
    # a hand-written shape does.
    return SimpleNamespace(
        data=DataConfigStub(
            index_path=index_path,
            validation_index_path=validation_index_path,
            split=SimpleNamespace(
                validation_fraction=validation_split,
                holdout_site=holdout_site,
                type=split_strategy,
            ),
        )
    )


# --------------------------------------------------------------------------- #
# read_manifest_records
# --------------------------------------------------------------------------- #
def test_read_manifest_records_records_wrapper(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    _write_manifest(p, [_rec("s0"), _rec("s1")])
    recs = read_manifest_records(p)
    assert recs is not None and len(recs) == 2
    assert {r["subject_id"] for r in recs} == {"s0", "s1"}


def test_read_manifest_records_bare_list(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps([_rec("s0")]))
    recs = read_manifest_records(p)
    assert recs is not None and recs[0]["subject_id"] == "s0"


def test_read_manifest_records_missing_returns_none(tmp_path: Path) -> None:
    assert read_manifest_records(tmp_path / "nope.json") is None


# --------------------------------------------------------------------------- #
# two-manifest (explicit) — the primary leak case
# --------------------------------------------------------------------------- #
def test_explicit_manifest_subject_overlap_is_leak(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    _write_manifest(train, [_rec("subjA"), _rec("subjB"), _rec("subjC")])
    _write_manifest(val, [_rec("subjC"), _rec("subjD")])  # subjC leaks
    rep = analyze_split_leakage(
        _cfg(index_path=str(train), validation_index_path=str(val))
    )
    assert isinstance(rep, SplitLeakageReport)
    assert rep.status == "leak"
    assert rep.mode == "explicit_manifest"
    assert rep.key_kind == "subject"
    assert "subjC" in rep.overlap


def test_explicit_manifest_disjoint_is_clean(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    _write_manifest(train, [_rec("Training_vol0"), _rec("Training_vol1")])
    # mrixfields-correct: val subjects are split-namespaced, never colliding.
    _write_manifest(val, [_rec("Validating_vol0"), _rec("Validating_vol1")])
    rep = analyze_split_leakage(
        _cfg(index_path=str(train), validation_index_path=str(val))
    )
    assert rep.status == "clean"
    assert rep.overlap == ()


def test_explicit_manifest_file_overlap_is_leak_even_if_subject_differs(
    tmp_path: Path,
) -> None:
    # Same physical file re-labelled with a different subject_id in val: the
    # subject sets are disjoint but the file identity collides — still a leak.
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    shared = _rec("Training_vol0")
    dup = dict(shared)
    dup["subject_id"] = "Validating_vol0"  # relabelled, same relative_path
    _write_manifest(train, [shared, _rec("Training_vol1")])
    _write_manifest(val, [dup])
    rep = analyze_split_leakage(
        _cfg(index_path=str(train), validation_index_path=str(val))
    )
    assert rep.status == "leak"
    assert rep.key_kind == "file"
    assert shared["relative_path"] in rep.overlap


def test_explicit_manifest_absent_files_skipped(tmp_path: Path) -> None:
    # Gitignored locally — must skip (not crash, not falsely pass).
    rep = analyze_split_leakage(
        _cfg(
            index_path=str(tmp_path / "train.json"),
            validation_index_path=str(tmp_path / "val.json"),
        )
    )
    assert rep.status == "skipped"


# --------------------------------------------------------------------------- #
# single-index random — support the random strategy
# --------------------------------------------------------------------------- #
def test_single_index_subject_straddle_is_leak(tmp_path: Path) -> None:
    # split_index holds out the LAST n_val records. A subject whose records
    # straddle that boundary leaks across train/val.
    idx = tmp_path / "idx.json"
    recs = [
        _rec("s0", field=1.5),
        _rec("s1", field=1.5),
        _rec("s2", field=1.5),
        _rec("s2", field=3.0),  # s2 also in the held-out tail
    ]
    _write_manifest(idx, recs)
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.25, split_strategy="random")
    )
    assert rep.status == "leak"
    assert rep.mode == "single_index"
    assert "s2" in rep.overlap


def test_single_index_clean_when_subjects_contiguous(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    recs = [_rec(f"s{i}") for i in range(10)]  # every subject unique
    _write_manifest(idx, recs)
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.2, split_strategy="random")
    )
    assert rep.status == "clean"


def test_single_index_train_only_split_is_clean(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    _write_manifest(idx, [_rec("s0"), _rec("s0")])  # duplicate subject, but no val
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.0, split_strategy="random")
    )
    assert rep.status == "clean"  # train-only: nothing to leak into


# --------------------------------------------------------------------------- #
# loso — support leave-one-site-out
# --------------------------------------------------------------------------- #
def test_loso_subject_across_sites_is_leak(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    a = _rec("patX")
    a["metadata"] = {"site": "siteA"}
    b = _rec("patX", field=7.0)
    b["metadata"] = {"site": "siteB"}  # same patient, holdout site
    c = _rec("patY")
    c["metadata"] = {"site": "siteA"}
    _write_manifest(idx, [a, b, c])
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), holdout_site="siteB", split_strategy="loso")
    )
    assert rep.status == "leak"
    assert rep.mode == "loso"
    assert "patX" in rep.overlap


def test_loso_disjoint_sites_clean(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    a = _rec("patX")
    a["metadata"] = {"site": "siteA"}
    b = _rec("patY")
    b["metadata"] = {"site": "siteB"}
    _write_manifest(idx, [a, b])
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), holdout_site="siteB", split_strategy="loso")
    )
    assert rep.status == "clean"


# --------------------------------------------------------------------------- #
# directory / auto — folder-disjoint by construction
# --------------------------------------------------------------------------- #
def test_directory_split_skipped(tmp_path: Path) -> None:
    rep = analyze_split_leakage(_cfg(split_strategy="directory"))
    assert rep.status == "skipped"
    assert rep.mode in {"directory", "unknown"}
