"""Paired-PNG super-resolution dataset (brats_sr: HR fully-sampled / LR undersampled).

``build_png_paired_index`` pairs an LR directory (``A_LRSI``) against an HR
directory (``A_HRSI``) by common filename — reusing the previously-orphaned
``find_common_png_stems`` / ``load_grayscale_png`` utils — and the dataset loads
each pair into a ``tio.Subject`` (input=LR, target=HR, optional lesion LabelMap).
"""
from __future__ import annotations

import numpy as np
import pytest

from mriforge.data.datasets.png_paired_dataset import (
    PngPairedDataset,
    build_png_paired_index,
)


def _png(path, value: int):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(path)


def _make_brats(root, lesion: bool = False):
    for name in ("img_001.png", "img_002.png"):
        _png(root / "A_LRSI" / name, 50)  # low-res input
        _png(root / "A_HRSI" / name, 200)  # high-res target
    _png(root / "A_HRSI" / "img_003.png", 200)  # HR-only → unpaired, excluded
    if lesion:
        _png(root / "Lesion_map" / "img_001.png", 1)  # only img_001 has a lesion


# ── indexer ───────────────────────────────────────────────────────────────────

def test_pairs_lr_and_hr_by_common_filename(tmp_path):
    _make_brats(tmp_path)
    recs = build_png_paired_index(tmp_path)
    assert sorted(r["stem"] for r in recs) == ["img_001.png", "img_002.png"]
    assert "A_LRSI" in recs[0]["input"]
    assert "A_HRSI" in recs[0]["target"]


def test_unpaired_hr_only_excluded(tmp_path):
    _make_brats(tmp_path)
    recs = build_png_paired_index(tmp_path)
    assert all(r["stem"] != "img_003.png" for r in recs)  # no fake pairing


def test_lesion_attached_when_present(tmp_path):
    _make_brats(tmp_path, lesion=True)
    recs = build_png_paired_index(tmp_path, lesion_dir="Lesion_map")
    by_stem = {r["stem"]: r for r in recs}
    assert by_stem["img_001.png"]["lesion"] is not None
    assert "Lesion_map" in by_stem["img_001.png"]["lesion"]
    assert by_stem["img_002.png"]["lesion"] is None  # no lesion for img_002


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_png_paired_index(tmp_path / "nope")


def test_missing_field_dir_raises(tmp_path):
    (tmp_path / "A_LRSI").mkdir()  # HR dir absent
    with pytest.raises(ValueError, match="dir"):
        build_png_paired_index(tmp_path)


# ── dataset ───────────────────────────────────────────────────────────────────

def test_dataset_getitem_returns_subject(tmp_path):
    import torchio as tio

    _make_brats(tmp_path)
    ds = PngPairedDataset(build_png_paired_index(tmp_path))
    assert len(ds) == 2
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    assert "input" in subj and "target" in subj
    # LR (50/255) is darker than HR (200/255)
    assert float(subj["input"].tensor.mean()) < float(subj["target"].tensor.mean())


def test_dataset_carries_lesion_labelmap(tmp_path):
    import torchio as tio

    _make_brats(tmp_path, lesion=True)
    ds = PngPairedDataset(build_png_paired_index(tmp_path, lesion_dir="Lesion_map"))
    subj = ds[0]  # img_001 (sorted first) → has lesion
    assert "lesion" in subj
    assert isinstance(subj["lesion"], tio.LabelMap)


# ── schema (data.png_paired) ──────────────────────────────────────────────────

def test_schema_defaults_disabled():
    from mriforge.config.schemas.data import DataConfigSchema, PngPairedConfigSchema

    assert PngPairedConfigSchema().enabled is False
    assert DataConfigSchema().png_paired.enabled is False


def test_schema_same_dir_raises():
    from pydantic import ValidationError

    from mriforge.config.schemas.data import PngPairedConfigSchema

    with pytest.raises(ValidationError, match="differ"):
        PngPairedConfigSchema(enabled=True, lr_dir="x", hr_dir="x")


def test_dry_iter_returns_length_correct_stub_subjects() -> None:
    """tio.Queue length probe works without a voxel read (the dry_iter contract).

    Regression for the '<Dataset>' object has no attribute 'dry_iter' crash
    class (oracle_bssfp / exp_vf_29, 2026-06): a SubjectsDataset wrapped in a
    tio.Queue must expose dry_iter yielding length-correct stub Subjects.
    """
    import torchio as tio

    from mriforge.data.datasets.png_paired_dataset import PngPairedDataset

    ds = PngPairedDataset.__new__(PngPairedDataset)
    ds.index = [{}, {}, {}]
    stubs = ds.dry_iter()
    assert len(stubs) == 3
    for s in stubs:
        assert isinstance(s, tio.Subject)
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)
