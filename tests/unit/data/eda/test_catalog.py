"""Tests for the manifest-driven dataset catalog."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mriforge.data.eda.catalog import (
    DatasetEntry,
    classify_modality,
    discover,
    resolve_scan_dir,
)


def _write_h5(path: Path) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=np.zeros((2, 8, 8), np.complex64))


def _stub_manifest(manifests: Path, name: str, data_root: str, **extra) -> Path:
    """An unpopulated external-style descriptor stub (empty records[])."""
    manifests.mkdir(parents=True, exist_ok=True)
    mpath = manifests / f"{name}.json"
    body = {
        "dataset_name": name,
        "data_root": data_root,
        "file_type": "h5",
        "records": [],
        "status": "not_downloaded",
        "source": {"role": "RAW", "raw_kspace": True, "anatomy": "brain"},
    }
    body.update(extra)
    mpath.write_text(json.dumps(body))
    return mpath


def test_discover_present_and_absent(present_kspace_manifest, absent_external_manifest, tmp_path):
    manifests_root = present_kspace_manifest.parent  # tmp_path/manifests (external is a subdir)
    entries = discover(manifests_root, data_root=tmp_path)
    by_id = {e.dataset_id: e for e in entries}
    assert "tiny_kspace" in by_id
    assert "calgary_campinas" in by_id
    assert by_id["tiny_kspace"].tier == "present"
    assert by_id["tiny_kspace"].modality == "kspace"
    assert by_id["calgary_campinas"].tier == "absent"
    assert by_id["calgary_campinas"].status == "not_downloaded"
    assert by_id["calgary_campinas"].source["role"] == "RAW"


def test_classify_modality_rules():
    assert classify_modality(role="RAW", raw_kspace=True, file_type="h5", name="m4raw_kspace") == "kspace"
    assert classify_modality(role="MAP", raw_kspace=False, file_type="nifti", name="nist_mrf") == "map"
    assert classify_modality(role="img", raw_kspace=False, file_type="png", name="ulf_paired_brain") == "paired"
    assert classify_modality(role="img", raw_kspace=False, file_type="nifti", name="mock") == "image"
    assert (
        classify_modality(role="RAW", raw_kspace=True, file_type="h5", name="kasper_7t_spiral_fmri")
        == "noncartesian"
    )


def test_discover_promotes_stub_with_on_disk_data(tmp_path):
    """A descriptor stub (empty records[], status=not_downloaded) whose data_root actually
    holds files on disk must render — trust the filesystem over an unpopulated manifest.
    This is the 2026-06-13 regression: a fully-downloaded external dataset was shown 'absent'
    only because its manifest stub was never populated on that machine."""
    db = tmp_path / "databases" / "external" / "osi2one_x" / "raw"
    _write_h5(db / "scan0.h5")
    mpath = _stub_manifest(
        tmp_path / "manifests" / "external", "osi2one_x",
        data_root="databases/external/osi2one_x/raw",
    )
    e = {x.dataset_id: x for x in discover(mpath.parent, data_root=tmp_path)}["osi2one_x"]
    assert e.tier == "present"
    assert e.status == "downloaded"  # promoted: it IS on disk despite the stub status


def test_discover_resolves_extracted_sibling_when_data_root_points_at_raw(tmp_path):
    """data_relocated: manifest data_root → '<x>/raw' (archives) while the unpacked data sits in
    the sibling '<x>/extracted'. The dataset must still render."""
    base = tmp_path / "databases" / "external" / "oracle_y"
    (base / "raw").mkdir(parents=True)             # archives only — no data ext
    _write_h5(base / "extracted" / "vol0.h5")      # real data lives here
    mpath = _stub_manifest(
        tmp_path / "manifests" / "external", "oracle_y",
        data_root="databases/external/oracle_y/raw",
    )
    e = {x.dataset_id: x for x in discover(mpath.parent, data_root=tmp_path)}["oracle_y"]
    assert e.tier == "present"


def test_discover_defective_status_stays_absent_even_with_disk_data(tmp_path):
    """A manifest explicitly flagged 'defective' is never auto-promoted, even if a stray data
    file sits under its tree (controlled-access sets like cine_055t_15t carry only sidecars)."""
    db = tmp_path / "databases" / "external" / "cine_z" / "raw"
    _write_h5(db / "sidecar.h5")
    mpath = _stub_manifest(
        tmp_path / "manifests" / "external", "cine_z",
        data_root="databases/external/cine_z/raw", status="defective",
    )
    e = {x.dataset_id: x for x in discover(mpath.parent, data_root=tmp_path)}["cine_z"]
    assert e.tier == "absent"


def test_discover_does_not_false_match_sibling_dataset_data(tmp_path):
    """The safety guard: a genuinely-absent dataset whose data_root does NOT exist must NOT walk
    up to the shared 'external/' container and false-match a *sibling's* data (the historical
    calgary→zenodo false-match). Absent stays absent."""
    ext = tmp_path / "databases" / "external"
    _write_h5(ext / "other_dataset" / "extracted" / "real.h5")  # a sibling's data
    mpath = _stub_manifest(
        tmp_path / "manifests" / "external", "calgary_campinas",
        data_root="databases/external/calgary_campinas/raw",  # this dir does NOT exist
    )
    e = {x.dataset_id: x for x in discover(mpath.parent, data_root=tmp_path)}["calgary_campinas"]
    assert e.tier == "absent"


def test_discover_derives_data_root_from_records_when_field_absent(tmp_path):
    """ulf_paired_brain_*_v5: data_root is None but records carry full 'primary_path' values.
    Derive the data_root from their common ancestor so the dataset renders instead of showing a
    'no scan dir' card."""
    root = tmp_path / "experiments" / "results" / "preprocessing" / "ulf" / "registered"
    _write_h5(root / "a.h5")
    _write_h5(root / "b.h5")
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True)
    mpath = manifests / "ulf_paired_brain_registered_v5.json"
    mpath.write_text(json.dumps({
        "dataset_name": "ulf_paired_brain_registered_v5",
        "data_root": None,
        "file_type": "h5",
        "records": [
            {"primary_path": str(root / "a.h5")},
            {"primary_path": str(root / "b.h5")},
        ],
    }))
    e = {x.dataset_id: x for x in discover(manifests, data_root=tmp_path)}[
        "ulf_paired_brain_registered_v5"
    ]
    assert e.data_root is not None
    assert Path(e.data_root) == root


def test_resolve_scan_dir_prefers_data_bearing_dir(tmp_path):
    """resolve_scan_dir: data_root itself wins when it holds data; otherwise the extracted
    sibling; a None or data-less input that resolves nowhere returns None/best-effort."""
    base = tmp_path / "databases" / "external" / "ds" / "raw"
    base.mkdir(parents=True)
    assert resolve_scan_dir(None) is None
    # raw empty, extracted has data → extracted
    _write_h5(base.parent / "extracted" / "v.h5")
    assert resolve_scan_dir(base) == base.parent / "extracted"


def test_classify_ismrmrd_raw_acquisition_is_noncartesian():
    """ISMRMRD/.mrd raw acquisitions are a list of readouts (n_acq, coil, samples) — a naive
    ifft2c fabricates a meaningless image, exactly like BART/TWIX. They must classify as
    'noncartesian' so the EDA shows the acquired-samples log-magnitude only (osi2one, mridata)."""
    assert classify_modality(role="RAW", raw_kspace=True, file_type="ismrmrd",
                             name="osi2one_47mt_recon") == "noncartesian"
    assert classify_modality(role="RAW", raw_kspace=True, file_type="mrd",
                             name="mridata_org") == "noncartesian"
    # a Cartesian-grid fastMRI/M4Raw h5 stays 'kspace' (proper recon)
    assert classify_modality(role="RAW", raw_kspace=True, file_type="h5",
                             name="m4raw_kspace") == "kspace"


def test_resolve_scan_dir_climbs_to_data_bearing_dataset_dir(tmp_path):
    """ULF image_* family: data_root points deep into a never-generated subpath
    (…/Data/64mT_data_image/isotropic) while the real data sits a few levels up at …/Data. The
    resolver must climb past the dataset-owned dirs to the nearest data-bearing ancestor — but
    STOP at a shared container so an absent set can't match a sibling."""
    data_dir = tmp_path / "databases" / "ulf_paired" / "ds64" / "Data"
    _write_h5(data_dir / "3T_data" / "sub-01" / "anat" / "img.h5")  # real data at …/Data
    missing = data_dir / "64mT_data_image" / "isotropic"            # declared, never generated
    assert resolve_scan_dir(missing) == data_dir


def test_resolve_scan_dir_stops_at_shared_container_for_absent_set(tmp_path):
    """A genuinely-absent external set (its own dir missing) must NOT climb into the shared
    external/ container and match a sibling's data — even with the deeper walk."""
    ext = tmp_path / "databases" / "external"
    _write_h5(ext / "sibling_ds" / "extracted" / "real.h5")
    missing = ext / "calgary_campinas" / "raw"   # calgary_campinas dir does not exist
    got = resolve_scan_dir(missing)
    assert got is None or "sibling_ds" not in str(got)


def test_derive_root_from_records_climbs_to_real_data(tmp_path):
    """isotropic_v5 (data_root=None) records carry primary_path values whose leaf
    (…/64mT_data_image/…/sub/anat/x.nii.gz) was never generated, but the real data is at …/Data.
    Discovery derives a root the resolver can climb from → tier=present."""
    import json as _json

    data_dir = tmp_path / "databases" / "ulf_paired" / "ds64" / "Data"
    _write_h5(data_dir / "3T_data" / "sub-01" / "anat" / "img.h5")
    rec_leaf = "databases/ulf_paired/ds64/Data/64mT_data_image/iso/sub-01/anat/sub-01_FLAIR.nii.gz"
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "ulf_paired_brain_isotropic_v5.json").write_text(_json.dumps({
        "dataset_name": "ulf_paired_brain_isotropic_v5", "data_root": None, "file_type": "nifti",
        "records": [{"primary_path": rec_leaf}, {"primary_path": rec_leaf.replace("sub-01", "sub-02")}],
    }))
    e = {x.dataset_id: x for x in discover(manifests, data_root=tmp_path)}["ulf_paired_brain_isotropic_v5"]
    assert e.tier == "present"
    # the real fix: the loader can now resolve a scan dir (climbs from the never-generated leaf)
    assert resolve_scan_dir(e.data_root) == data_dir


def test_discover_physical_descends_into_external_for_uncovered_data(tmp_path):
    """Data downloaded under external/<x>/ whose manifest is stale or mis-named (kirby_kki vs
    kirby_kki_multimodal) was invisible: discover_physical skipped external/ wholesale. It must
    descend one level and surface uncovered data dirs as disk_<name> — while web-placeholder dirs
    (openneuro_hub: only openneuro.org) and manifest-covered dirs are NOT surfaced."""
    from mriforge.data.eda.catalog import discover_physical

    db = tmp_path / "databases"
    _write_h5(db / "external" / "kirby_kki" / "extracted" / "scan.h5")     # uncovered real data
    (db / "external" / "openneuro_hub" / "raw").mkdir(parents=True)
    (db / "external" / "openneuro_hub" / "raw" / "openneuro.org").write_text("x")  # web placeholder
    _write_h5(db / "external" / "oracle_bssfp" / "extracted" / "v.h5")     # covered by a manifest
    covered = [
        DatasetEntry("oracle_bssfp", (), db, db / "external" / "oracle_bssfp" / "extracted",
                     "h5", (), "kspace", "present", "downloaded", {}),
    ]
    ids = {e.dataset_id for e in discover_physical(db, covered)}
    assert "disk_kirby_kki" in ids          # uncovered external data → surfaced
    assert "disk_openneuro_hub" not in ids   # web placeholder (no data ext) → not surfaced
    assert "disk_oracle_bssfp" not in ids    # already covered by a manifest → not surfaced


def test_resolve_scan_dir_prefers_real_data_over_archive_only(tmp_path):
    """mrf_100mt_optimum regression: raw/ holds only a .zip (archive) while extracted/ holds the
    real .npy dictionary. resolve_scan_dir must prefer the real-data dir, not the archive-only
    one it happens to hit first."""
    base = tmp_path / "databases" / "external" / "mrf"
    (base / "raw").mkdir(parents=True)
    (base / "raw" / "dict.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # empty zip (archive)
    _write_h5(base / "extracted" / "parameterTable.h5")                     # the real data
    assert resolve_scan_dir(base / "raw") == base / "extracted"
