"""Unit tests for the MRIxFields2026 paired full-volume eval loader.

Covers:
- Happy-path pairing: one (source, target) pair yielded correctly.
- Volume shape, dtype, [0,1] range, voxel_size, and path attributes.
- Missing-side skip: if a subject has source but no target (or vice versa),
  the subject is skipped and a warning is logged.
- Out-of-range volume: still yielded but a range warning is logged.
- Split filter: records whose relative_path does not start with the
  requested split prefix are excluded.
"""
from __future__ import annotations

import json
import logging

import nibabel as nib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPLIT = "Validating_prospective"


def _write_nifti(path, data: np.ndarray) -> None:
    """Write a float32 NIfTI volume (identity affine, 1 mm isotropic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(path))


def _make_challenge_data(tmp_path, records):
    """Write NIfTI files referenced by records and return data_root."""
    data_root = tmp_path / "ChallengeData"
    for rec in records:
        fpath = data_root / rec["relative_path"]
        vol = np.linspace(0, 1, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
        _write_nifti(fpath, vol)
    return data_root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def paired_records(tmp_path):
    """Two records for subject s0001 at 0.1T and 7T, contrast T1w."""
    records = [
        {
            "relative_path": f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_0001.nii.gz",
            "field_strength": 0.1,
            "contrast": "T1w",
            "subject_id": "s0001",
            "pairing_group": "g0001",
            "file_id": "f0001",
            "shape": [8, 8, 4],
        },
        {
            "relative_path": f"{_SPLIT}/T1W/7T/P_T1W_7T_0001.nii.gz",
            "field_strength": 7.0,
            "contrast": "T1w",
            "subject_id": "s0001",
            "pairing_group": "g0001",
            "file_id": "f0002",
            "shape": [8, 8, 4],
        },
    ]
    data_root = _make_challenge_data(tmp_path, records)
    return records, data_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIterPairedVolumes:

    def test_happy_path_yields_one_pair(self, paired_records):
        """Default (ordinal) pairing yields the single rank-0 pair."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = paired_records
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0, contrast="T1w"
            )
        )
        assert len(results) == 1

    def test_paired_volume_attributes(self, paired_records):
        """subject_id-mode pairing: attributes + shared-id subject_id preserved."""
        from spectramr.data.builders.mrixfields_baseline_eval import PairedVolume, iter_paired_volumes

        records, data_root = paired_records
        pv = next(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0, contrast="T1w",
                pairing="subject_id",
            )
        )
        assert isinstance(pv, PairedVolume)
        # Shape
        assert pv.source.shape == (8, 8, 4)
        assert pv.target.shape == (8, 8, 4)
        # dtype
        assert pv.source.dtype == np.float32
        assert pv.target.dtype == np.float32
        # [0,1] range
        assert float(pv.source.min()) >= 0.0
        assert float(pv.source.max()) <= 1.0
        assert float(pv.target.min()) >= 0.0
        assert float(pv.target.max()) <= 1.0
        # voxel_size
        assert pv.voxel_size == (1.0, 1.0, 1.0)
        # paths resolve to the correct NIfTI
        assert pv.source_path.name.endswith("0.1T_0001.nii.gz")
        assert pv.target_path.name.endswith("7T_0001.nii.gz")
        # subject_id
        assert pv.subject_id == "s0001"
        # affine is a 4x4 array
        assert pv.affine.shape == (4, 4)

    def test_missing_target_skips_and_warns(self, tmp_path, caplog):
        """subject_id mode: subject with source but no target is skipped + warns."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records = [
            {
                "relative_path": f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_0002.nii.gz",
                "field_strength": 0.1,
                "contrast": "T1w",
                "subject_id": "s0002",
                "pairing_group": "g0002",
                "file_id": "f0003",
                "shape": [8, 8, 4],
            }
            # No 7T record for s0002
        ]
        data_root = _make_challenge_data(tmp_path, records)

        with caplog.at_level(logging.WARNING, logger="spectramr.data.builders.mrixfields_baseline_eval"):
            results = list(
                iter_paired_volumes(
                    records, data_root, source_field=0.1, target_field=7.0, contrast="T1w",
                    pairing="subject_id",
                )
            )

        assert results == []
        assert any("missing" in msg.lower() or "skipping" in msg.lower() for msg in caplog.messages)

    def test_missing_source_skips_and_warns(self, tmp_path, caplog):
        """subject_id mode: subject with target but no source is skipped + warns."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records = [
            {
                "relative_path": f"{_SPLIT}/T1W/7T/P_T1W_7T_0004.nii.gz",
                "field_strength": 7.0,
                "contrast": "T1w",
                "subject_id": "s0004",
                "pairing_group": "g0004",
                "file_id": "f0006",
                "shape": [8, 8, 4],
            }
            # No 0.1T source record for s0004
        ]
        data_root = _make_challenge_data(tmp_path, records)

        with caplog.at_level(logging.WARNING, logger="spectramr.data.builders.mrixfields_baseline_eval"):
            results = list(
                iter_paired_volumes(
                    records, data_root, source_field=0.1, target_field=7.0, contrast="T1w",
                    pairing="subject_id",
                )
            )

        assert results == []
        assert any("missing" in msg.lower() or "skipping" in msg.lower() for msg in caplog.messages)

    def test_out_of_range_volume_still_yielded_with_warning(self, tmp_path, caplog):
        """A volume with values up to 5.0 is yielded but a range warning is logged."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        # Write one record with a volume that has values in [0, 5]
        records = [
            {
                "relative_path": f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_0003.nii.gz",
                "field_strength": 0.1,
                "contrast": "T1w",
                "subject_id": "s0003",
                "pairing_group": "g0003",
                "file_id": "f0004",
                "shape": [8, 8, 4],
            },
            {
                "relative_path": f"{_SPLIT}/T1W/7T/P_T1W_7T_0003.nii.gz",
                "field_strength": 7.0,
                "contrast": "T1w",
                "subject_id": "s0003",
                "pairing_group": "g0003",
                "file_id": "f0005",
                "shape": [8, 8, 4],
            },
        ]
        data_root = tmp_path / "ChallengeData"
        # Source: values in [0, 5] — out of [0,1]
        src_path = data_root / records[0]["relative_path"]
        src_path.parent.mkdir(parents=True, exist_ok=True)
        vol_oor = np.linspace(0, 5, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
        img_oor = nib.Nifti1Image(vol_oor, np.eye(4))
        img_oor.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img_oor, str(src_path))
        # Target: normal [0,1]
        tgt_path = data_root / records[1]["relative_path"]
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        vol_ok = np.linspace(0, 1, 8 * 8 * 4, dtype=np.float32).reshape(8, 8, 4)
        img_ok = nib.Nifti1Image(vol_ok, np.eye(4))
        img_ok.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img_ok, str(tgt_path))

        with caplog.at_level(logging.WARNING, logger="spectramr.data.builders.mrixfields_baseline_eval"):
            results = list(
                iter_paired_volumes(
                    records, data_root, source_field=0.1, target_field=7.0, contrast="T1w"
                )
            )

        # Volume is still yielded
        assert len(results) == 1
        # ...and NOT rescaled: the out-of-range values survive unchanged.
        assert float(results[0].source.max()) > 1.5
        # A warning about out-of-range was logged
        assert any("outside" in msg.lower() or "range" in msg.lower() or "max" in msg.lower()
                   for msg in caplog.messages)

    def test_split_filter_excludes_other_splits(self, tmp_path):
        """Records from a different split prefix are excluded."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records = [
            {
                "relative_path": "Training/T1W/0.1T/P_T1W_0.1T_0001.nii.gz",
                "field_strength": 0.1,
                "contrast": "T1w",
                "subject_id": "s0001",
                "pairing_group": "g0001",
                "file_id": "f9001",
                "shape": [8, 8, 4],
            },
            {
                "relative_path": "Training/T1W/7T/P_T1W_7T_0001.nii.gz",
                "field_strength": 7.0,
                "contrast": "T1w",
                "subject_id": "s0001",
                "pairing_group": "g0001",
                "file_id": "f9002",
                "shape": [8, 8, 4],
            },
        ]
        # Do NOT write any files — they should never be loaded
        data_root = tmp_path / "ChallengeData"
        data_root.mkdir(parents=True, exist_ok=True)

        results = list(
            iter_paired_volumes(
                records, data_root,
                source_field=0.1, target_field=7.0, contrast="T1w",
                split="Validating_prospective",
            )
        )
        assert results == []

    def test_contrast_filter_excludes_wrong_contrast(self, paired_records):
        """Records with a different contrast are excluded."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = paired_records
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0, contrast="T2w"
            )
        )
        assert results == []


class TestOrdinalPairing:
    """Ordinal (rank-within-field) pairing — the travelling-volunteer default."""

    @staticmethod
    def _cross_field_records(tmp_path, src_ids, tgt_ids):
        """Write source (0.1T) + target (7T) T1w volumes with DISTINCT subject ids.

        Mirrors the val manifest, whose subject_id numbering is per-field so source
        and target ids never coincide (id-matching would yield zero pairs).
        """
        records = []
        for sid in src_ids:
            rel = f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_{sid}.nii.gz"
            records.append(
                {"relative_path": rel, "field_strength": 0.1, "contrast": "T1w",
                 "subject_id": sid}
            )
        for tid in tgt_ids:
            rel = f"{_SPLIT}/T1W/7T/P_T1W_7T_{tid}.nii.gz"
            records.append(
                {"relative_path": rel, "field_strength": 7.0, "contrast": "T1w",
                 "subject_id": tid}
            )
        data_root = _make_challenge_data(tmp_path, records)
        return records, data_root

    def test_ordinal_pairs_distinct_ids_by_rank(self, tmp_path):
        """Source ids {0001,0002} <-> target ids {0016,0017} pair 2 by rank."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(
            tmp_path, ["0001", "0002"], ["0016", "0017"]
        )
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0,
                contrast="T1w", pairing="ordinal",
            )
        )
        assert len(results) == 2
        # rank pairing: 0001<->0016, 0002<->0017 (subject_id records the pairing)
        assert results[0].subject_id == "0001->0016"
        assert results[1].subject_id == "0002->0017"
        assert results[0].source_path.name.endswith("0.1T_0001.nii.gz")
        assert results[0].target_path.name.endswith("7T_0016.nii.gz")

    def test_ordinal_default_is_ordinal(self, tmp_path):
        """Omitting `pairing` uses ordinal (distinct ids still pair — proves default)."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(
            tmp_path, ["0001", "0002"], ["0016", "0017"]
        )
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0, contrast="T1w"
            )
        )
        assert len(results) == 2
        assert results[0].subject_id == "0001->0016"

    def test_subject_id_mode_yields_zero_on_distinct_ids(self, tmp_path):
        """subject_id pairing yields NOTHING on per-field ids (the C1 bug it fixes)."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(
            tmp_path, ["0001", "0002"], ["0016", "0017"]
        )
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0,
                contrast="T1w", pairing="subject_id",
            )
        )
        assert results == []

    def test_ordinal_count_mismatch_warns(self, tmp_path, caplog):
        """Unequal source/target counts pair min(len) and warn naming the counts."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(
            tmp_path, ["0001", "0002"], ["0016"]  # 2 source, 1 target
        )
        with caplog.at_level(
            logging.WARNING, logger="spectramr.data.builders.mrixfields_baseline_eval"
        ):
            results = list(
                iter_paired_volumes(
                    records, data_root, source_field=0.1, target_field=7.0,
                    contrast="T1w", pairing="ordinal",
                )
            )
        assert len(results) == 1
        assert results[0].subject_id == "0001->0016"
        # warning names both counts (2 source vs 1 target)
        assert any(
            "mismatch" in m.lower() and "2" in m and "1" in m for m in caplog.messages
        )

    def test_ordinal_empty_side_yields_nothing(self, tmp_path):
        """Empty target-field list yields nothing (no warning; runner guard catches)."""
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(
            tmp_path, ["0001", "0002"], []  # no target records
        )
        results = list(
            iter_paired_volumes(
                records, data_root, source_field=0.1, target_field=7.0,
                contrast="T1w", pairing="ordinal",
            )
        )
        assert results == []

    def test_unknown_pairing_raises(self, tmp_path):
        from spectramr.data.builders.mrixfields_baseline_eval import iter_paired_volumes

        records, data_root = self._cross_field_records(tmp_path, ["0001"], ["0016"])
        with pytest.raises(ValueError, match="pairing"):
            list(
                iter_paired_volumes(
                    records, data_root, source_field=0.1, target_field=7.0,
                    contrast="T1w", pairing="bogus",
                )
            )


class TestLoadManifestRecords:

    def test_load_manifest_records(self, tmp_path):
        from spectramr.data.builders.mrixfields_baseline_eval import load_manifest_records

        manifest = {
            "records": [
                {
                    "relative_path": f"{_SPLIT}/T1W/0.1T/P_T1W_0.1T_0001.nii.gz",
                    "field_strength": 0.1,
                    "contrast": "T1w",
                    "subject_id": "s0001",
                    "pairing_group": "g0001",
                    "file_id": "f0001",
                    "shape": [8, 8, 4],
                }
            ]
        }
        manifest_path = tmp_path / "mrixfields2026_val.json"
        manifest_path.write_text(json.dumps(manifest))

        records = load_manifest_records(manifest_path)
        assert len(records) == 1
        assert records[0]["contrast"] == "T1w"
        assert records[0]["subject_id"] == "s0001"
