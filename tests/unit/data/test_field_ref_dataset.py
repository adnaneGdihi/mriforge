"""NIfTI field-reference dataset — pair anatomy with a real B0/B1 map.

For datasets that ship a real field map as a NIfTI (e.g. kasper's
``b0MapSmoothed..._Hz.nii``, traveling_heads' B1+ maps): the anatomy is the VF
target and the field rides as ``b0_map`` / ``b1_map`` for the real-reference
field-scoring seam. The derivation-free counterpart to BART b0_from_echo /
b1_from_dam.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mriforge.data.datasets.field_ref_dataset import (
    FieldRefDataset,
    build_field_ref_index,
)
from tests.utils.data_config_stub import DataConfigStub


def _nii(path, val, shape=(4, 4, 2)):
    nib = pytest.importorskip("nibabel")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full(shape, val, np.float32), np.eye(4)), str(path))


def _make_kasper_like(root):
    _nii(root / "recon" / "me1.nii", 1.0)
    _nii(root / "recon" / "fmri1.nii", 2.0)
    _nii(root / "maps" / "b0_Hz.nii", 50.0)


# ── indexer ───────────────────────────────────────────────────────────────────

def test_build_index_pairs_each_anatomy_with_the_field_map(tmp_path):
    _make_kasper_like(tmp_path)
    recs = build_field_ref_index(
        tmp_path, anatomy_glob="recon/*.nii", b0_map="maps/b0_Hz.nii"
    )
    assert len(recs) == 2  # me1 + fmri1
    assert all(r["b0_map"].endswith("b0_Hz.nii") for r in recs)
    assert {Path(r["target"]).stem for r in recs} == {"me1", "fmri1"}


def test_requires_at_least_one_field_map(tmp_path):
    _make_kasper_like(tmp_path)
    with pytest.raises(ValueError, match="b0_map / b1_map"):
        build_field_ref_index(tmp_path, anatomy_glob="recon/*.nii")


def test_missing_field_map_file_raises(tmp_path):
    _make_kasper_like(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_field_ref_index(tmp_path, anatomy_glob="recon/*.nii", b0_map="maps/nope.nii")


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_field_ref_index(tmp_path / "nope", anatomy_glob="*.nii", b0_map="b0.nii")


def test_no_anatomy_raises(tmp_path):
    _nii(tmp_path / "maps" / "b0_Hz.nii", 50.0)
    with pytest.raises(FileNotFoundError, match="anatomy"):
        build_field_ref_index(tmp_path, anatomy_glob="recon/*.nii", b0_map="maps/b0_Hz.nii")


# ── dataset ───────────────────────────────────────────────────────────────────

def test_dataset_emits_target_and_b0_map(tmp_path):
    pytest.importorskip("nibabel")
    import torchio as tio

    _make_kasper_like(tmp_path)
    ds = FieldRefDataset(
        build_field_ref_index(tmp_path, anatomy_glob="recon/*.nii", b0_map="maps/b0_Hz.nii")
    )
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    assert "target" in subj and "b0_map" in subj
    assert "input" in subj  # twin degrades input downstream
    # the B0 map carries the real value (50 Hz)
    assert abs(float(subj["b0_map"].tensor.mean()) - 50.0) < 1e-3


# ── schema + dispatch ─────────────────────────────────────────────────────────

def test_schema_defaults_disabled_and_requires_a_field_map():
    from pydantic import ValidationError

    from mriforge.config.schemas.data import DataConfigSchema, FieldRefConfigSchema

    assert FieldRefConfigSchema().enabled is False
    assert DataConfigSchema().field_ref.enabled is False
    with pytest.raises(ValidationError, match="b0_map"):
        FieldRefConfigSchema(enabled=True)  # neither b0_map nor b1_map


def test_field_ref_is_an_accepted_dataset_type():
    from mriforge.config.schemas.data import DataConfigSchema

    assert DataConfigSchema(dataset_type="field_ref").dataset_type == "field_ref"


def test_dispatch_requires_enabled():
    from types import SimpleNamespace

    from mriforge.data.builders.dataset_instantiator import DatasetInstantiator

    config = DataConfigStub(
        dataset_type="field_ref",
        num_virtual_coils=4,
        index_path=None,
        field_ref=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match=r"data\.field_ref\.enabled"):
        DatasetInstantiator.create_datasets(
            config, train_index=[], val_index=[], train_transforms=None, val_transforms=None
        )


def test_dry_iter_returns_length_correct_stub_subjects() -> None:
    """tio.Queue length probe works without a voxel read (the dry_iter contract).

    Regression for the '<Dataset>' object has no attribute 'dry_iter' crash
    class (oracle_bssfp / exp_vf_29, 2026-06): a SubjectsDataset wrapped in a
    tio.Queue must expose dry_iter yielding length-correct stub Subjects.
    """
    import torchio as tio

    from mriforge.data.datasets.field_ref_dataset import FieldRefDataset

    ds = FieldRefDataset.__new__(FieldRefDataset)
    ds.index = [{}, {}, {}]
    stubs = ds.dry_iter()
    assert len(stubs) == 3
    for s in stubs:
        assert isinstance(s, tio.Subject)
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)
