"""ISMRMRD measured-trajectory dataset (kasper monitored spiral → vf_35).

Reconstructs from raw ISMRMRD acquisitions using the file's OWN measured
trajectory (not a synthesized golden-angle one), via the existing NUFFT/DCF
physics. The synthetic .mrd carries a normalized [-0.5, 0.5] trajectory so the
default 2π scale maps it to the torchkbnufft [-π, π] convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from mriforge.config.schemas.data import IsmrmrdConfigSchema
from mriforge.data.datasets.ismrmrd_dataset import (
    IsmrmrdKspaceDataset,
    build_ismrmrd_index,
)


def _write_mrd(path, n_acq=12, n_coil=2, n_samp=16, traj_dim=2):
    import h5py

    head_dt = np.dtype(
        [
            ("number_of_samples", "<u2"),
            ("active_channels", "<u2"),
            ("trajectory_dimensions", "<u2"),
        ]
    )
    vlen = h5py.vlen_dtype(np.float32)
    acq_dt = np.dtype([("head", head_dt), ("traj", vlen), ("data", vlen)])
    arr = np.empty(n_acq, dtype=acq_dt)
    rng = np.random.default_rng(0)
    for i in range(n_acq):
        arr["head"][i] = (n_samp, n_coil, traj_dim)
        arr["data"][i] = rng.standard_normal(2 * n_coil * n_samp).astype(np.float32)
        # normalized [-0.5, 0.5] trajectory (ISMRMRD convention)
        arr["traj"][i] = (rng.random(traj_dim * n_samp).astype(np.float32) - 0.5)
    xml = (
        '<?xml version="1.0"?>'
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
        "<encoding><trajectory>spiral</trajectory>"
        "<encodedSpace><matrixSize><x>16</x><y>16</y><z>1</z></matrixSize>"
        "</encodedSpace></encoding></ismrmrdHeader>"
    )
    with h5py.File(path, "w") as f:
        g = f.create_group("dataset")
        g.create_dataset("data", data=arr)
        g.create_dataset("xml", data=np.array([xml], dtype=object),
                         dtype=h5py.string_dtype())


def test_build_index_lists_ismrmrd_files(tmp_path):
    _write_mrd(tmp_path / "vol1.h5")
    _write_mrd(tmp_path / "vol2.mrd")
    (tmp_path / "notes.txt").write_text("x")
    idx = build_ismrmrd_index(tmp_path)
    assert len(idx) == 2  # .h5 + .mrd, not .txt


def test_recon_uses_measured_trajectory(tmp_path):
    pytest.importorskip("torchkbnufft")
    import torchio as tio

    _write_mrd(tmp_path / "spiral.h5")
    cfg = IsmrmrdConfigSchema(
        enabled=True, im_size=[16, 16], density_compensation="iterative"
    )
    ds = IsmrmrdKspaceDataset(build_ismrmrd_index(tmp_path), cfg)
    assert len(ds) == 1
    subj = ds[0]
    assert isinstance(subj, tio.Subject)
    assert "input" in subj and "target" in subj and "kspace" in subj
    recon = subj["target"].tensor  # (C, H, W, 1)
    assert recon.shape[-3:] == (16, 16, 1)
    assert recon.std() > 0  # mechanism fires — non-degenerate recon


def test_im_size_falls_back_to_header(tmp_path):
    pytest.importorskip("torchkbnufft")
    _write_mrd(tmp_path / "spiral.h5")
    cfg = IsmrmrdConfigSchema(enabled=True, im_size=None)  # → header matrixSize 16
    subj = IsmrmrdKspaceDataset(build_ismrmrd_index(tmp_path), cfg)[0]
    assert subj["target"].tensor.shape[-3:] == (16, 16, 1)


def test_requires_enabled(tmp_path):
    with pytest.raises(ValueError, match="enabled"):
        IsmrmrdKspaceDataset([], IsmrmrdConfigSchema(enabled=False))


def test_schema_defaults_and_allow_list():
    from mriforge.config.schemas.data import DataConfigSchema

    assert DataConfigSchema().ismrmrd.enabled is False
    assert DataConfigSchema().ismrmrd.density_compensation == "iterative"
    assert (
        DataConfigSchema(dataset_type="ismrmrd_kspace").dataset_type == "ismrmrd_kspace"
    )


# ── G2: paired nominal + measured trajectory emit (vf_35) ──────────────────


def test_schema_paired_trajectory_fields():
    cfg = IsmrmrdConfigSchema(
        enabled=True, emit_paired_trajectory=True, nominal_file_glob="*nominal*"
    )
    assert cfg.emit_paired_trajectory is True
    assert cfg.nominal_file_glob == "*nominal*"
    # emit without a glob must raise (no silent no-op — pitfall #15)
    with pytest.raises(Exception):
        IsmrmrdConfigSchema(enabled=True, emit_paired_trajectory=True)


def test_resolve_nominal_file(tmp_path):
    from mriforge.data.datasets.ismrmrd_dataset import resolve_nominal_file

    (tmp_path / "spiral_monitored.h5").write_text("x")
    nom = tmp_path / "spiral_nominal.h5"
    nom.write_text("y")
    measured = str(tmp_path / "spiral_monitored.h5")
    assert resolve_nominal_file(measured, "*nominal*") == str(nom)
    # no sibling match → raise (no silent fallback)
    with pytest.raises(FileNotFoundError):
        resolve_nominal_file(measured, "*does_not_exist*")


def test_build_index_excludes_nominal(tmp_path):
    _write_mrd(tmp_path / "spiral_monitored.h5")
    _write_mrd(tmp_path / "spiral_nominal.h5")
    idx = build_ismrmrd_index(tmp_path, exclude_glob="*nominal*")
    assert len(idx) == 1 and "monitored" in idx[0]


def test_emit_paired_trajectory(tmp_path):
    pytest.importorskip("torchkbnufft")
    _write_mrd(tmp_path / "spiral_monitored.h5")
    _write_mrd(tmp_path / "spiral_nominal.h5")
    cfg = IsmrmrdConfigSchema(
        enabled=True,
        im_size=[16, 16],
        emit_paired_trajectory=True,
        nominal_file_glob="*nominal*",
    )
    idx = build_ismrmrd_index(tmp_path, exclude_glob="*nominal*")
    subj = IsmrmrdKspaceDataset(idx, cfg)[0]
    assert "trajectory_measured" in subj and "trajectory_nominal" in subj
    assert subj["trajectory_measured"].shape == subj["trajectory_nominal"].shape


def test_dry_iter_returns_length_correct_stub_subjects() -> None:
    """tio.Queue length probe works without a voxel read (the dry_iter contract).

    Regression for the '<Dataset>' object has no attribute 'dry_iter' crash
    class (oracle_bssfp / exp_vf_29, 2026-06): a SubjectsDataset wrapped in a
    tio.Queue must expose dry_iter yielding length-correct stub Subjects.
    """
    import torchio as tio

    from mriforge.data.datasets.ismrmrd_dataset import IsmrmrdKspaceDataset

    ds = IsmrmrdKspaceDataset.__new__(IsmrmrdKspaceDataset)
    ds.index = ['a', 'b', 'c']
    stubs = ds.dry_iter()
    assert len(stubs) == 3
    for s in stubs:
        assert isinstance(s, tio.Subject)
        assert tuple(s["input"].tensor.shape) == (1, 1, 1, 1)
