"""Tests for the measured dataset profiler (``mriforge.data.profiler``).

Phase 1 covers the pure logic that closes the measured-capability gap (pitfall
#18): the canonical physics-feature → need map, ``derive_needs`` three-state
behaviour, and ``crosscheck_needs`` (which normalizes the asserted ``source.needs``
— a list that may mix ints and descriptive strings like ``["phase", 10]`` — against
the measured needs and treats ``unknown`` as "not contradicted").
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from mriforge.data import profiler

# --- canonical feature <-> need map -------------------------------------------------


def test_need_to_feature_covers_the_physics_feature_needs() -> None:
    """The seven feature-mapped needs map 1:1 onto the canonical physics keys."""
    assert profiler.NEED_TO_FEATURE == {
        1: "b0_field",
        2: "b1_plus",
        3: "phase_cycled_bssfp",
        4: "trajectory",
        5: "mrf",
        6: "temporal_4d",
        7: "qmri",
    }
    # every mapped feature is a recognised physics-feature key
    for feat in profiler.NEED_TO_FEATURE.values():
        assert feat in profiler.PHYSICS_FEATURE_KEYS


def test_feature_constructors_are_three_state() -> None:
    assert profiler.present("complex dtype") == {
        "state": "present",
        "evidence": "complex dtype",
    }
    assert profiler.absent() == {"state": "absent", "evidence": None}
    assert profiler.unknown("no header") == {"state": "unknown", "evidence": "no header"}


# --- derive_needs -------------------------------------------------------------------


def _profile(**features) -> dict:
    """A minimal profile dict with the given physics features (rest absent)."""
    feats = {k: profiler.absent() for k in profiler.PHYSICS_FEATURE_KEYS}
    feats.update(features)
    return {
        "inventory": {"file_types": {}, "vendors": []},
        "geometry": {"n_coils": None},
        "contrast": {"field_strength_t": None},
        "physics_features": feats,
    }


def test_derive_needs_present_features_become_present_needs() -> None:
    prof = _profile(b1_plus=profiler.present("b1map"), qmri=profiler.present("t1map"))
    prof["geometry"]["n_coils"] = 4
    out = profiler.derive_needs(prof)
    assert out["present"] == [2, 7, 10]  # b1+, qmri, multi-coil
    assert 4 not in out["present"]  # trajectory absent


def test_derive_needs_unknown_feature_is_unknown_not_present() -> None:
    prof = _profile(b0_field=profiler.unknown("bart hdr has no TE"))
    out = profiler.derive_needs(prof)
    assert 1 not in out["present"]
    assert 1 in out["unknown"]


def test_derive_needs_multicoil_three_state() -> None:
    present = _profile()
    present["geometry"]["n_coils"] = 8
    assert 10 in profiler.derive_needs(present)["present"]

    single = _profile()
    single["geometry"]["n_coils"] = 1
    out = profiler.derive_needs(single)
    assert 10 not in out["present"] and 10 not in out["unknown"]  # provably absent

    nocoil = _profile()  # n_coils None
    assert 10 in profiler.derive_needs(nocoil)["unknown"]


def test_derive_needs_ulf_from_field_strength() -> None:
    ulf = _profile()
    ulf["contrast"]["field_strength_t"] = 0.064
    assert 8 in profiler.derive_needs(ulf)["present"]

    midfield = _profile()
    midfield["contrast"]["field_strength_t"] = 0.3  # M4Raw — not ULF
    out = profiler.derive_needs(midfield)
    assert 8 not in out["present"] and 8 not in out["unknown"]

    unknown_field = _profile()  # field None
    assert 8 in profiler.derive_needs(unknown_field)["unknown"]


def test_derive_needs_multivendor() -> None:
    multi = _profile()
    multi["inventory"]["vendors"] = ["Siemens", "GE"]
    assert 9 in profiler.derive_needs(multi)["present"]

    single = _profile()
    single["inventory"]["vendors"] = ["Siemens"]
    out = profiler.derive_needs(single)
    assert 9 not in out["present"] and 9 not in out["unknown"]


# --- crosscheck_needs ---------------------------------------------------------------


def test_crosscheck_agree_with_extra_measured_need() -> None:
    """Asserting fewer needs than measured is agreement (data over-provides)."""
    out = profiler.crosscheck_needs(asserted=[7], derived_present=[7, 10], derived_unknown=[])
    assert out["agree"] is True
    assert out["extra_in_data"] == [10]
    assert out["missing_from_data"] == []


def test_crosscheck_flags_provably_absent_claimed_need() -> None:
    """A claimed need the data provably lacks is the dangerous mismatch (pitfall #18)."""
    out = profiler.crosscheck_needs(asserted=[2], derived_present=[], derived_unknown=[])
    assert out["agree"] is False
    assert out["missing_from_data"] == [2]


def test_crosscheck_unknown_is_not_contradicted() -> None:
    """A claimed need whose feature could not be measured is NOT flagged (no false alarm)."""
    out = profiler.crosscheck_needs(asserted=[2], derived_present=[], derived_unknown=[2])
    assert out["agree"] is True
    assert out["missing_from_data"] == []


def test_crosscheck_normalizes_mixed_int_and_string_needs() -> None:
    """``source.needs`` may be ``["phase", 10, "7"]`` — coerce numerics, keep the rest."""
    out = profiler.crosscheck_needs(
        asserted=["phase", 10, "7"], derived_present=[7, 10], derived_unknown=[]
    )
    assert out["asserted"] == [7, 10]  # numeric, sorted, deduped
    assert out["non_numeric"] == ["phase"]
    assert out["agree"] is True


# ============================ Phase 2: per-format extractors ============================


def _write_bart(base, dims, payload=True):
    """Write a minimal BART .hdr (+ optional .cfl) pair at ``base`` (no suffix)."""
    base.with_suffix(".hdr").write_text("# Dimensions\n" + " ".join(map(str, dims)) + "\n")
    if payload:
        n = int(np.prod(dims))
        base.with_suffix(".cfl").write_bytes(struct.pack(f"<{2 * n}f", *([0.0] * (2 * n))))


def test_profile_bart_reads_dims_and_is_complex(tmp_path):
    base = tmp_path / "kspace"
    _write_bart(base, [256, 256, 16, 4])
    rec = profiler.profile_bart(base.with_suffix(".cfl"))
    assert rec["reader"] == "bart-hdr"
    assert rec["geometry"]["matrix"] == [256, 256]
    assert rec["geometry"]["n_slices"] == 16
    assert rec["geometry"]["domain"] == "kspace"
    # BART .hdr cannot disambiguate coil/echo/time dims -> honest unknown, not a guess.
    assert rec["geometry"]["n_coils"] is None
    # raw k-space is complex -> phase is provably present.
    assert rec["features"]["phase"]["state"] == profiler.PRESENT


def test_profile_bart_detects_trajectory_sidecar(tmp_path):
    base = tmp_path / "spiral"
    _write_bart(base, [512, 64, 1, 8])
    _write_bart(tmp_path / "spiral_traj", [3, 512, 64], payload=True)
    rec = profiler.profile_bart(base.with_suffix(".cfl"))
    assert rec["features"]["trajectory"]["state"] == profiler.PRESENT
    assert "traj" in rec["features"]["trajectory"]["evidence"].lower()


def test_profile_bart_filename_reveals_qmri(tmp_path):
    base = tmp_path / "t1map"
    _write_bart(base, [128, 128])
    rec = profiler.profile_bart(base.with_suffix(".cfl"))
    assert rec["features"]["qmri"]["state"] == profiler.PRESENT


def test_profile_nifti_reads_spacing_and_4d_is_temporal(tmp_path):
    import nibabel as nib

    vol3d = tmp_path / "anat.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((192, 192, 20), dtype=np.float32), np.diag([1.1, 1.1, 4.0, 1.0])),
        vol3d,
    )
    rec = profiler.profile_nifti(vol3d)
    assert rec["reader"] == "nifti-header"
    assert rec["geometry"]["matrix"] == [192, 192]
    assert rec["geometry"]["voxel_spacing_mm"][:2] == [pytest.approx(1.1), pytest.approx(1.1)]
    assert rec["geometry"]["domain"] == "image"
    assert rec["features"]["temporal_4d"]["state"] == profiler.ABSENT

    vol4d = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 30, 100), dtype=np.float32), np.eye(4)), vol4d)
    rec4 = profiler.profile_nifti(vol4d)
    assert rec4["features"]["temporal_4d"]["state"] == profiler.PRESENT


_ISMRMRD_XML = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <acquisitionSystemInformation>
    <systemVendor>{vendor}</systemVendor>
    <systemFieldStrength_T>{field}</systemFieldStrength_T>
    <receiverChannels>{coils}</receiverChannels>
  </acquisitionSystemInformation>
  <encoding>
    <trajectory>{traj}</trajectory>
    <encodedSpace><matrixSize><x>{mx}</x><y>{my}</y><z>{mz}</z></matrixSize>
      <fieldOfView_mm><x>{fx}</x><y>{fy}</y><z>{fz}</z></fieldOfView_mm></encodedSpace>
  </encoding>
  <sequenceParameters>{te}</sequenceParameters>
</ismrmrdHeader>"""


def _write_ismrmrd(path, *, vendor="Siemens", field=3.0, coils=32, traj="spiral", te=(2.0,)):
    import h5py

    te_xml = "".join(f"<TE>{t}</TE>" for t in te)
    xml = _ISMRMRD_XML.format(
        vendor=vendor, field=field, coils=coils, traj=traj,
        mx=256, my=256, mz=1, fx=230, fy=230, fz=5, te=te_xml,
    )
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset")
        g.create_dataset("xml", data=np.array([xml.encode()]))


def test_profile_ismrmrd_reads_header_and_trajectory(tmp_path):
    mrd = tmp_path / "spiral.mrd"
    _write_ismrmrd(mrd, traj="spiral", coils=32, field=7.0)
    rec = profiler.profile_ismrmrd(mrd)
    assert rec["reader"] == "ismrmrd-xml"
    assert rec["geometry"]["n_coils"] == 32
    assert rec["geometry"]["matrix"] == [256, 256]
    assert rec["contrast"]["field_strength_t"] == pytest.approx(7.0)
    assert rec["vendor"] == "Siemens"
    assert rec["features"]["trajectory"]["state"] == profiler.PRESENT  # spiral != cartesian


def test_profile_ismrmrd_cartesian_trajectory_is_absent(tmp_path):
    mrd = tmp_path / "cart.mrd"
    _write_ismrmrd(mrd, traj="cartesian")
    rec = profiler.profile_ismrmrd(mrd)
    assert rec["features"]["trajectory"]["state"] == profiler.ABSENT


def test_profile_ismrmrd_multi_echo_reveals_b0(tmp_path):
    mrd = tmp_path / "multiecho.mrd"
    _write_ismrmrd(mrd, traj="cartesian", te=(2.3, 4.6, 6.9))
    rec = profiler.profile_ismrmrd(mrd)
    assert rec["features"]["b0_field"]["state"] == profiler.PRESENT
    assert rec["contrast"]["n_echoes"] == 3


def test_profile_dicom_reads_sequence_params(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ds = Dataset()
    ds.MagneticFieldStrength = 1.5
    ds.RepetitionTime = 500.0
    ds.EchoTime = 12.0
    ds.FlipAngle = 70.0
    ds.Manufacturer = "GE MEDICAL SYSTEMS"
    ds.Rows = 256
    ds.Columns = 256
    ds.PixelSpacing = [0.9, 0.9]
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    p = tmp_path / "slice.dcm"
    pydicom.dcmwrite(p, ds, enforce_file_format=True)

    rec = profiler.profile_dicom(p)
    assert rec["reader"] == "dicom-tags"
    assert rec["contrast"]["field_strength_t"] == pytest.approx(1.5)
    assert rec["contrast"]["tr_ms"] == [pytest.approx(500.0)]
    assert rec["contrast"]["te_ms"] == [pytest.approx(12.0)]
    assert rec["vendor"] == "GE MEDICAL SYSTEMS"
    assert rec["geometry"]["matrix"] == [256, 256]


def test_profile_record_dispatches_siemens_ima_to_dicom(tmp_path):
    """A Siemens ``.IMA`` record dispatches to the DICOM reader. The extension is
    uppercase as Siemens writes it, so the suffix dispatch must be case-insensitive
    (the St. Jude NIST-phantom ingest, 2026-06-11)."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    ds = Dataset()
    ds.MagneticFieldStrength = 3.0
    ds.Manufacturer = "SIEMENS"
    ds.Rows = 256
    ds.Columns = 256
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    p = tmp_path / "STJUDE_Siemens_SKYRA_3T_08132015.IMA"
    pydicom.dcmwrite(p, ds, enforce_file_format=True)

    # suffix dispatch (lower-cased) routes the uppercase .IMA to the dicom reader
    assert profiler._SUFFIX_DISPATCH[".ima"] == "dicom"
    assert profiler.profile_record(p)["reader"] == "dicom-tags"


def _write_twix(path, *, field=2.89362, coils=4, base=128, pe=128, contrasts=1):
    """Write a minimal Siemens TWIX VD ``.dat`` with a header + one scan header (coils)."""
    asc = (
        f'<ParamLong."iMaxNoOfRxChannels">  {{ {coils}  }}\n'
        "### ASCCONV BEGIN object=x version=1 ###\n"
        f"sProtConsistencyInfo.flNominalB0\t = \t{field}\n"
        'tSequenceFileName\t = \t"%SiemensSeq%\\trufi"\n'
        f"sKSpace.lBaseResolution\t = \t{base}\n"
        f"sKSpace.lPhaseEncodingLines\t = \t{pe}\n"
        f"lContrasts\t = \t{contrasts}\n"
        "### ASCCONV END ###\n"
    ).encode("latin-1")
    scan = bytearray(192)
    struct.pack_into("<H", scan, 48, 8)  # ushSamplesInScan
    struct.pack_into("<H", scan, 50, coils)  # ushUsedChannels
    hdr_len = 4 + len(asc)
    body = struct.pack("<I", hdr_len) + asc + bytes(scan)
    offset = 8 + 152
    entry = struct.pack("<IIQQ64s64s", 1, 1, offset, len(body), b"p", b"p")
    path.write_bytes(struct.pack("<II", 0, 1) + entry + body)


def test_profile_twix_reads_protocol_header(tmp_path):
    p = tmp_path / "meas_MID00018.dat"
    _write_twix(p, field=0.55, coils=4, base=96, pe=96, contrasts=1)
    rec = profiler.profile_twix(p)
    assert rec["reader"] == "twix-header"
    assert rec["vendor"] == "Siemens"
    assert rec["geometry"]["n_coils"] == 4
    assert rec["geometry"]["matrix"] == [96, 96]
    assert rec["geometry"]["domain"] == "kspace"
    assert rec["contrast"]["field_strength_t"] == pytest.approx(0.55)
    assert rec["contrast"]["sequence"] == "trufi"
    assert rec["features"]["phase"]["state"] == profiler.PRESENT
    # a single header cannot prove RF phase-cycling -> left UNKNOWN, never guessed absent
    assert "phase_cycled_bssfp" not in rec["features"]


def test_profile_twix_multi_contrast_reveals_b0(tmp_path):
    p = tmp_path / "multiecho.dat"
    _write_twix(p, contrasts=3)
    rec = profiler.profile_twix(p)
    assert rec["features"]["b0_field"]["state"] == profiler.PRESENT
    assert rec["contrast"]["n_echoes"] == 3


def test_profile_record_dispatches_dat_to_twix(tmp_path):
    p = tmp_path / "scan.dat"
    _write_twix(p)
    assert profiler.profile_record(p, file_type="twix")["reader"] == "twix-header"
    assert profiler.profile_record(p)["reader"] == "twix-header"  # suffix dispatch


def test_profile_dataset_twix_derives_multicoil_and_ulf(tmp_path):
    """A 64 mT 8-coil TWIX dataset measures needs 8 (ULF) and 10 (multi-coil)."""
    root = tmp_path / "raw"
    root.mkdir()
    _write_twix(root / "s1.dat", field=0.064, coils=8)
    _write_twix(root / "s2.dat", field=0.064, coils=8)
    records = [
        {"relative_path": "s1.dat", "file_id": "s1"},
        {"relative_path": "s2.dat", "file_id": "s2"},
    ]
    block = profiler.profile_dataset(root, "twix", records, asserted_needs=[8, 10])
    assert 10 in block["derived_needs"]  # 8 coils > 1
    assert 8 in block["derived_needs"]  # 0.064 T <= 0.1 T ULF threshold
    assert block["inventory"]["vendors"] == ["Siemens"]
    assert block["needs_crosscheck"]["agree"] is True


def test_profile_record_dispatches_by_suffix(tmp_path):
    base = tmp_path / "k"
    _write_bart(base, [64, 64])
    rec = profiler.profile_record(base.with_suffix(".cfl"))
    assert rec["reader"] == "bart-hdr"


# ============================ Phase 3: dataset aggregator ============================


def test_aggregate_features_precedence_present_over_absent_over_unknown() -> None:
    """present beats absent beats unknown; geometry/vendors merge across records."""
    recs = [
        {
            "geometry": {"n_coils": 4, "matrix": [256, 256], "domain": "kspace"},
            "contrast": {"field_strength_t": 3.0},
            "features": {"trajectory": profiler.absent(), "phase": profiler.present("c")},
            "vendor": "Siemens",
        },
        {
            "geometry": {"n_coils": 4, "matrix": [256, 256], "domain": "kspace"},
            "contrast": {"field_strength_t": 3.0},
            "features": {"trajectory": profiler.present("spiral"), "b0_field": profiler.absent()},
            "vendor": "GE",
        },
    ]
    agg = profiler.aggregate_profiles(recs)
    assert agg["physics_features"]["trajectory"]["state"] == profiler.PRESENT  # present wins
    assert agg["physics_features"]["b0_field"]["state"] == profiler.ABSENT
    assert agg["physics_features"]["b1_plus"]["state"] == profiler.UNKNOWN  # never observed
    assert agg["geometry"]["n_coils"] == 4
    assert sorted(agg["vendors"]) == ["GE", "Siemens"]


def test_profile_dataset_bart_end_to_end(tmp_path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    _write_bart(root / "t1map", [128, 128, 8])
    _write_bart(root / "scan2", [128, 128, 8])
    records = [
        {"relative_path": "t1map", "file_id": "t1map"},
        {"relative_path": "scan2", "file_id": "scan2"},
    ]
    block = profiler.profile_dataset(
        root, "bart", records, asserted_needs=[7], stamp="STAMP", host="HOST"
    )
    assert block["physics_features"]["qmri"]["state"] == profiler.PRESENT  # t1map filename
    assert block["physics_features"]["phase"]["state"] == profiler.PRESENT
    assert 7 in block["derived_needs"]
    assert 10 not in block["derived_needs"]  # BART n_coils unknown -> not multi-coil
    assert block["needs_crosscheck"]["agree"] is True
    assert block["inventory"]["n_records_profiled"] == 2
    assert block["inventory"]["file_types"] == {".cfl": 2, ".hdr": 2}
    assert block["provenance"]["profiled_at"] == "STAMP"
    assert block["geometry"]["matrix"] == [128, 128]


def test_profile_dataset_flags_unsupported_claim(tmp_path) -> None:
    """A dataset asserting B1+ whose files provably lack it is flagged (pitfall #18)."""
    root = tmp_path / "raw"
    root.mkdir()
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.zeros((64, 64, 10), dtype=np.float32), np.eye(4)), root / "anat.nii.gz")
    records = [{"relative_path": "anat.nii.gz", "file_id": "anat"}]
    block = profiler.profile_dataset(root, "nifti", records, asserted_needs=[2], stamp="S", host="H")
    # NIfTI can't observe B1+; it is unknown (not contradicted) -> NOT a hard miss.
    assert block["needs_crosscheck"]["agree"] is True
    assert block["physics_features"]["temporal_4d"]["state"] == profiler.ABSENT  # 3D


def test_profile_dataset_skips_unreadable_records(tmp_path) -> None:
    """A corrupt/missing record is skipped, not fatal; n_records_profiled reflects it."""
    root = tmp_path / "raw"
    root.mkdir()
    _write_bart(root / "good", [64, 64])
    records = [
        {"relative_path": "good", "file_id": "good"},
        {"relative_path": "missing", "file_id": "missing"},  # no .hdr -> skipped
    ]
    block = profiler.profile_dataset(root, "bart", records, asserted_needs=[], stamp="S", host="H")
    assert block["inventory"]["n_records_profiled"] == 1
    assert block["inventory"]["n_records_skipped"] == 1


# ============================ Phase 4: crosscheck a committed manifest ============================


def test_crosscheck_manifest_recomputes_from_profile_block() -> None:
    manifest = {
        "source": {"needs": [7, "phase"]},
        "profile": {"derived_needs": [7, 10], "unknown_needs": [1]},
    }
    cc = profiler.crosscheck_manifest(manifest)
    assert cc is not None
    assert cc["agree"] is True  # 7 measured; nothing claimed-but-absent
    assert cc["extra_in_data"] == [10]


def test_crosscheck_manifest_flags_committed_mismatch() -> None:
    manifest = {
        "source": {"needs": [2]},  # claims B1+
        "profile": {"derived_needs": [10], "unknown_needs": []},  # data has only multi-coil
    }
    cc = profiler.crosscheck_manifest(manifest)
    assert cc is not None and cc["agree"] is False
    assert cc["missing_from_data"] == [2]


def test_crosscheck_manifest_none_without_profile() -> None:
    assert profiler.crosscheck_manifest({"source": {"needs": [1]}}) is None
