"""Spec E5 — BIDS low-field/high-field paired indexer + dataset (ulf_paired).

The pure ``build_bids_paired_index`` pairs 64mT↔3T NIfTI by (subject, contrast)
across a BIDS tree; the dataset loads each pair into a ``tio.Subject``
(input=low-field, target=high-field). The indexer tests use synthetic empty
files (no NIfTI dep); the dataset test writes real NIfTI via nibabel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mriforge.data.datasets.bids_paired_dataset import (
    BidsPairedDataset,
    build_bids_paired_index,
    parse_bids_entities,
)


def _make_bids(root: Path, lo_contrasts=("T1w", "T2w"), hi_contrasts=("T1w", "T2w", "FLAIR")):
    for sub in ("sub-01", "sub-02"):
        hi = root / "Data" / "3T_data" / sub / "anat"
        hi.mkdir(parents=True)
        for c in hi_contrasts:
            (hi / f"{sub}_acq-highres_{c}.nii.gz").write_bytes(b"\x00")
        lo = root / "Data" / "64mT_data" / sub / "anat"
        lo.mkdir(parents=True)
        for c in lo_contrasts:
            (lo / f"{sub}_{c}.nii.gz").write_bytes(b"\x00")


# ── pure: parse_bids_entities ─────────────────────────────────────────────────

def test_parse_bids_entities_extracts_subject_and_suffix():
    ent = parse_bids_entities("sub-0011_acq-highres_T1w.nii.gz")
    assert ent["subject"] == "sub-0011"
    assert ent["suffix"] == "T1w"


def test_parse_bids_entities_with_session():
    ent = parse_bids_entities("sub-0018_ses-01_run-1_ADC.nii.gz")
    assert ent["subject"] == "sub-0018"
    assert ent["suffix"] == "ADC"


def test_parse_bids_entities_non_bids_returns_none_subject():
    assert parse_bids_entities("random.nii.gz")["subject"] is None


# ── indexer ───────────────────────────────────────────────────────────────────

def test_pairs_by_subject_and_contrast(tmp_path):
    _make_bids(tmp_path)
    recs = build_bids_paired_index(tmp_path)
    pairs = {(r["subject"], r["contrast"]) for r in recs}
    assert pairs == {
        ("sub-01", "T1w"),
        ("sub-01", "T2w"),
        ("sub-02", "T1w"),
        ("sub-02", "T2w"),
    }


def test_input_is_low_field_target_is_high_field(tmp_path):
    _make_bids(tmp_path)
    rec = build_bids_paired_index(tmp_path)[0]
    assert "64mT_data" in rec["input"]
    assert "3T_data" in rec["target"]


def test_unpaired_contrast_excluded(tmp_path):
    # FLAIR exists only in 3T → no 64mT partner → excluded (no fake pairing)
    _make_bids(tmp_path)
    recs = build_bids_paired_index(tmp_path)
    assert all(r["contrast"] != "FLAIR" for r in recs)


def test_contrasts_filter(tmp_path):
    _make_bids(tmp_path)
    recs = build_bids_paired_index(tmp_path, contrasts=("T1w",))
    assert {r["contrast"] for r in recs} == {"T1w"}


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_bids_paired_index(tmp_path / "nope")


def test_no_field_dirs_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="field"):
        build_bids_paired_index(tmp_path)


# ── dataset (real NIfTI via nibabel) ──────────────────────────────────────────

def test_dataset_getitem_returns_subject(tmp_path):
    nib = pytest.importorskip("nibabel")
    import torchio as tio

    def _nii(path, val):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((4, 4, 2), val, np.float32), np.eye(4)), str(path))

    _nii(tmp_path / "Data" / "3T_data" / "sub-01" / "anat" / "sub-01_acq-highres_T1w.nii.gz", 2.0)
    _nii(tmp_path / "Data" / "64mT_data" / "sub-01" / "anat" / "sub-01_T1w.nii.gz", 1.0)

    index = build_bids_paired_index(tmp_path)
    ds = BidsPairedDataset(index)
    assert len(ds) == 1
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    assert "input" in subj and "target" in subj


# ── schema (spec E5) ──────────────────────────────────────────────────────────

def test_bids_schema_defaults_disabled():
    from mriforge.config.schemas.data import BidsPairedConfigSchema, DataConfigSchema

    assert BidsPairedConfigSchema().enabled is False
    assert DataConfigSchema().bids_paired.enabled is False


def test_bids_schema_same_dir_raises():
    from pydantic import ValidationError

    from mriforge.config.schemas.data import BidsPairedConfigSchema

    with pytest.raises(ValidationError, match="differ"):
        BidsPairedConfigSchema(enabled=True, low_field_dir="x", high_field_dir="x")


def test_bids_schema_empty_contrasts_raises():
    from pydantic import ValidationError

    from mriforge.config.schemas.data import BidsPairedConfigSchema

    with pytest.raises(ValidationError, match="contrasts"):
        BidsPairedConfigSchema(enabled=True, contrasts=[])


# ── session/run pairing (regression — 2026-06-12 data-loader rigor audit) ──────


def _nii_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # indexer only stats names, never reads voxels


def test_parse_bids_entities_extracts_session_run_acq() -> None:
    """``ses``/``run``/``acq`` must be parsed, not silently ignored."""
    ent = parse_bids_entities("sub-01_ses-02_acq-highres_run-3_T1w.nii.gz")
    assert ent["subject"] == "sub-01"
    assert ent["suffix"] == "T1w"
    assert ent["ses"] == "02"
    assert ent["run"] == "3"
    assert ent["acq"] == "highres"


def test_bids_index_raises_on_multiple_acquisitions_per_field(tmp_path) -> None:
    """Two sessions for one (subject, contrast) within a field must raise.

    Regression for the silent ``setdefault`` drop: when a field held more than
    one acquisition per (subject, contrast) the indexer kept only the first and
    silently discarded the rest (pitfall #9). Cross-field ses/run/acq labels are
    allowed to differ, but within a field the pair must be unambiguous.
    """
    for ses in ("01", "02"):
        _nii_empty(
            tmp_path / "Data" / "64mT_data" / "sub-01" / f"ses-{ses}" / "anat"
            / f"sub-01_ses-{ses}_T1w.nii.gz"
        )
    _nii_empty(
        tmp_path / "Data" / "3T_data" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    )
    with pytest.raises(ValueError, match=r"[Aa]mbiguous"):
        build_bids_paired_index(tmp_path)


def test_bids_index_pairs_across_fields_with_differing_acq(tmp_path) -> None:
    """A low-field scan pairs with a high-field scan of the same (subject,
    contrast) even when their acq/ses labels differ (separate scanners)."""
    _nii_empty(
        tmp_path / "Data" / "64mT_data" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    )
    _nii_empty(
        tmp_path / "Data" / "3T_data" / "sub-01" / "anat"
        / "sub-01_acq-highres_T1w.nii.gz"
    )
    records = build_bids_paired_index(tmp_path)
    assert len(records) == 1
    assert records[0]["subject"] == "sub-01" and records[0]["contrast"] == "T1w"


def test_dry_iter_returns_length_correct_stub_subjects() -> None:
    """tio.Queue length probe works without a voxel read (the dry_iter contract).

    Regression for the '<Dataset>' object has no attribute 'dry_iter' crash
    class (oracle_bssfp / exp_vf_29, 2026-06): a SubjectsDataset wrapped in a
    tio.Queue must expose dry_iter yielding length-correct stub Subjects.
    """
    import torchio as tio

    from mriforge.data.datasets.bids_paired_dataset import BidsPairedDataset

    ds = BidsPairedDataset.__new__(BidsPairedDataset)
    ds.index = [{}, {}, {}]
    stubs = ds.dry_iter()
    assert len(stubs) == 3
    for s in stubs:
        assert isinstance(s, tio.Subject)
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)
