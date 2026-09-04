"""Tests for the manifest-vs-physical-tree dataset audit (:mod:`spectramr.data.eda.layout_audit`).

The audit reconciles each manifest's *declared* location against a ``tree -J`` snapshot of the
real ``databases/`` directory. These tests pin the verdict for every physical state — present,
extracted-elsewhere, archive-only, metadata-only, empty, absent — and the conservative matching
that stops a genuinely-absent dataset from being mis-located onto a shared container directory.
"""
from __future__ import annotations

import json

import pytest

from spectramr.data.eda.layout_audit import (
    AuditResult,
    audit_entry,
    audit_manifests,
    build_tree_index,
    classify_ext,
    format_markdown,
    proposed_fix,
    tree_relative,
)


# --- tiny tree -J builders ----------------------------------------------------------------
def _d(name, *children):
    return {"type": "directory", "name": name, "contents": list(children)}


def _f(name):
    return {"type": "file", "name": name}


def _tree(*root_children):
    return [_d(".", *root_children), {"type": "report", "directories": 0, "files": 0}]


@pytest.fixture
def index():
    """One index exercising every physical state the auditor must distinguish."""
    return build_tree_index(_tree(
        _d("external",
           _d("present_raw", _d("raw", _f("a.h5"), _f("b.h5"))),                       # data in raw → ok
           _d("extracted_only", _d("raw", _f("x.zip")),
              _d("extracted", _d("sub", _f("d.nii.gz")))),                              # data in extracted → relocated
           _d("archive_only", _d("raw", _f("only.zip"), _f("more.7z"))),               # archives, never unpacked
           _d("meta_only", _d("raw", _f("index.html?group_id=313"), _f("readme.json"))),  # failed/placeholder
           _d("empty_ds", _d("raw")),                                                  # dir exists, no files
           # 'absent_ds' deliberately not present
           ),
        _d("siteA", _d("uniquename", _f("u.h5"))),   # single named dir with data → locatable
        _d("siteB", _d("foo", _f("i.dcm"))),         # 'foo' appears twice with data → ambiguous
        _d("siteC", _d("foo", _f("j.dcm"))),
    ))


# --- extension classification -------------------------------------------------------------
@pytest.mark.parametrize("name,kind", [
    ("scan.h5", "data"), ("vol.nii.gz", "data"), ("im.dcm", "data"), ("k.cfl", "data"),
    ("IM.4983353", "data"),                       # numeric ext → DICOM frame / split file
    ("bundle.zip", "archive"), ("pack.tar.xz", "archive"), ("a.7z", "archive"),
    ("notes.json", "other"), ("header.hdr", "other"), ("README", "other"),
    ("index.html?group_id=313", "other"),
])
def test_classify_ext(name, kind):
    assert classify_ext(name) == kind


# --- tree_relative ------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("databases/external/x/raw", "external/x/raw"),
    ("/project/alpha_lab/researcher/spectramr/databases/m4raw/y", "m4raw/y"),
    ("external/x", "external/x"),
    (None, None),
])
def test_tree_relative(raw, expected):
    assert tree_relative(raw) == expected


# --- per-state verdicts -------------------------------------------------------------------
def test_present_data_at_declared_root_is_ok(index):
    res = audit_entry(dataset_id="present_raw", declared_root="external/present_raw/raw",
                      status="downloaded", index=index)
    assert res.verdict == "ok"
    assert res.data_files == 2 and res.status_flag is None


def test_data_in_extracted_is_relocated_not_archive(index):
    res = audit_entry(dataset_id="extracted_only", declared_root="external/extracted_only/raw",
                      status="not_downloaded", index=index)
    assert res.verdict == "data_relocated"
    assert res.found_at == "external/extracted_only/extracted/sub"
    assert res.status_flag == "present_but_marked_not_downloaded"


def test_archive_only_when_nothing_extracted(index):
    res = audit_entry(dataset_id="archive_only", declared_root="external/archive_only/raw",
                      status="downloaded", index=index)
    assert res.verdict == "archive_only"
    assert res.archive_files == 2
    assert res.status_flag == "claims_downloaded_but_no_data"


def test_metadata_only_is_failed_download(index):
    res = audit_entry(dataset_id="meta_only", declared_root="external/meta_only/raw",
                      status="downloaded", index=index)
    assert res.verdict == "metadata_only"


def test_empty_dir(index):
    res = audit_entry(dataset_id="empty_ds", declared_root="external/empty_ds/raw",
                      status="downloaded", index=index)
    assert res.verdict == "empty"


def test_absent_does_not_fall_back_to_shared_container(index):
    """The regression: a missing ``external/<x>/raw`` must NOT anchor on the shared ``external/``
    container (which holds data) and report a bogus relocation — it is genuinely absent."""
    res = audit_entry(dataset_id="calgary_campinas", declared_root="external/calgary_campinas/raw",
                      status="not_downloaded", index=index)
    assert res.verdict == "absent"
    assert res.found_at is None


def test_unresolvable_when_no_data_root(index):
    res = audit_entry(dataset_id="brokenptr", declared_root=None, status="downloaded", index=index)
    assert res.verdict == "unresolvable"
    assert res.status_flag == "claims_downloaded_but_no_data"


def test_relocation_by_unique_name(index):
    res = audit_entry(dataset_id="uniquename", declared_root="external/uniquename/raw",
                      status="downloaded", index=index)
    assert res.verdict == "data_relocated"
    assert res.found_at == "siteA/uniquename"


def test_ambiguous_name_is_not_guessed(index):
    """A dataset name shared by two data dirs must NOT be auto-located (no guessing)."""
    res = audit_entry(dataset_id="foo", declared_root="external/foo/raw",
                      status="downloaded", index=index)
    assert res.verdict == "absent"
    assert res.found_at is None


def test_missing_records_are_counted(index):
    res = audit_entry(
        dataset_id="present_raw", declared_root="external/present_raw/raw", status="downloaded",
        records=[{"relative_path": "a.h5"}, {"relative_path": "ghost.h5"}], index=index,
    )
    assert res.verdict == "ok"
    assert res.records_checked == 2 and res.records_missing == 1
    assert res.is_problem  # an ok dataset with unresolved records still needs attention


# --- full sweep + reverse uncovered check -------------------------------------------------
def test_audit_manifests_and_uncovered(tmp_path):
    repo = tmp_path
    manifests = repo / "data" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "present.json").write_text(json.dumps({
        "dataset_name": "present", "data_root": "databases/external/present/raw",
        "file_type": "h5", "records": [{"relative_path": "a.h5"}], "status": "not_downloaded",
    }))
    (manifests / "gone.json").write_text(json.dumps({
        "dataset_name": "gone", "data_root": "databases/external/gone/raw",
        "file_type": "h5", "records": [], "status": "not_downloaded",
    }))

    index = build_tree_index(_tree(
        _d("external", _d("present", _d("raw", _f("a.h5")))),
        _d("orphan_set", _d("scans", _f("o.h5"))),   # real data, no covering manifest
    ))
    report = audit_manifests(manifests, repo, index)
    by_id = {r.dataset_id: r for r in report.results}
    assert by_id["present"].verdict == "ok"
    assert by_id["present"].status_flag == "present_but_marked_not_downloaded"
    assert by_id["gone"].verdict == "absent"

    uncovered = {r.declared_root for r in report.uncovered}
    assert "orphan_set" in uncovered
    assert all(r.verdict == "no_manifest" for r in report.uncovered)


# --- auto-fix proposal (the --apply path) -------------------------------------------------
@pytest.fixture
def fix_index():
    return build_tree_index(_tree(
        _d("external",
           _d("ds_ok", _d("raw", _f("a.h5"), _f("b.h5"))),                          # data in raw
           _d("ds_ext", _d("raw", _f("x.zip")), _d("extracted", _d("s", _f("d.nii.gz")))),  # in extracted
           ),
        _d("core", _d("real", _f("c.h5"))),   # a core dataset a duplicate stub might point at
    ))


def test_proposed_fix_external_ok_populates_records(fix_index):
    r = audit_entry(dataset_id="ds_ok", declared_root="external/ds_ok/raw", status="not_downloaded",
                    index=fix_index, manifest_path="/x/data/manifests/external/ds_ok.json")
    fix = proposed_fix(r, fix_index)
    assert fix["data_root"] == "databases/external/ds_ok/raw"
    assert fix["status"] == "downloaded" and fix["total_records"] == 2
    assert {rec["relative_path"] for rec in fix["records"]} == {"a.h5", "b.h5"}


def test_proposed_fix_repoints_raw_to_extracted(fix_index):
    r = audit_entry(dataset_id="ds_ext", declared_root="external/ds_ext/raw", status="not_downloaded",
                    index=fix_index, manifest_path="/x/data/manifests/external/ds_ext.json")
    fix = proposed_fix(r, fix_index)
    assert fix["data_root"] == "databases/external/ds_ext/extracted"
    assert fix["records"] == [{"relative_path": "s/d.nii.gz", "file_id": "d.nii"}]


def test_proposed_fix_skips_duplicate_stub_pointing_at_core(fix_index):
    """An external stub whose data relocates OUTSIDE its own external dir (onto a core dataset)
    must not be auto-filled — that would point one manifest at another dataset's data."""
    r = AuditResult(dataset_id="dup", manifest_path="/x/data/manifests/external/dup.json",
                    declared_root="core/real", status="downloaded", verdict="data_relocated",
                    found_at="core/real")
    assert proposed_fix(r, fix_index) is None


def test_proposed_fix_skips_non_external(fix_index):
    r = audit_entry(dataset_id="ds_ok", declared_root="external/ds_ok/raw", status="downloaded",
                    index=fix_index, manifest_path="/x/data/manifests/m4raw_thing.json")
    assert proposed_fix(r, fix_index) is None


def test_format_markdown_groups_by_verdict(index):
    res = [
        audit_entry(dataset_id="present_raw", declared_root="external/present_raw/raw",
                    status="downloaded", index=index),
        audit_entry(dataset_id="calgary", declared_root="external/calgary/raw",
                    status="not_downloaded", index=index),
    ]
    from spectramr.data.eda.layout_audit import AuditReport
    md = format_markdown(AuditReport(results=res))
    assert "# Dataset layout audit" in md
    assert "ABSENT" in md and "OK" in md
