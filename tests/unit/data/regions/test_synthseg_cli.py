"""Tests for the CLI-based SynthSeg segmenter (WS4 core).

FreeSurfer is not available in CI, so ``subprocess.run`` is mocked to emit a stub
label volume. These assert the NIfTI round-trip (write -> call -> read), the
volumetric Dice, and the fail-loud behaviour (no silent fallback).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mriforge.data.nifti_export import VoxelGeometry
from mriforge.data.regions import synthseg_cli


def _geometry(n_slices: int, rows: int, cols: int) -> VoxelGeometry:
    return VoxelGeometry(
        slice_mm=5.0,
        row_mm=1.0,
        col_mm=1.0,
        n_slices=n_slices,
        rows=rows,
        cols=cols,
        slice_gap_assumed_zero=True,
        source="test",
    )


def _fake_synthseg(*, label: int = 10):
    """Return a subprocess.run stand-in that writes a stub label volume to --o."""

    def _run(argv, **_kw):
        import nibabel as nib

        i = Path(argv[argv.index("--i") + 1])
        o = Path(argv[argv.index("--o") + 1])
        vol = nib.load(str(i)).get_fdata()
        lab = np.zeros_like(vol, dtype=np.float32)
        sl = tuple(slice(0, max(1, s // 2)) for s in lab.shape)
        lab[sl] = label
        nib.save(nib.Nifti1Image(lab, np.eye(4)), str(o))
        return subprocess.CompletedProcess(argv, 0, "", "")

    return _run


def test_segment_volume_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(synthseg_cli.subprocess, "run", _fake_synthseg(label=10))
    vol = torch.rand(4, 16, 16)
    seg = synthseg_cli.segment_volume_via_synthseg(
        vol, _geometry(4, 16, 16), workdir=tmp_path
    )
    assert seg.shape == (4, 16, 16)
    assert seg.dtype == torch.int64
    assert (seg == 10).any()


def test_synthseg_dice_finite(monkeypatch, tmp_path):
    monkeypatch.setattr(synthseg_cli.subprocess, "run", _fake_synthseg(label=10))
    vol = torch.rand(4, 16, 16)
    d = synthseg_cli.synthseg_dice(
        vol, vol, _geometry(4, 16, 16), workdir=tmp_path, labels=(10,)
    )
    assert 0.0 <= d <= 1.0
    assert d == pytest.approx(1.0)  # identical stub labels -> perfect Dice


def test_raises_on_cli_failure(monkeypatch, tmp_path):
    def _fail(argv, **_kw):
        return subprocess.CompletedProcess(argv, 1, "", "boom")

    monkeypatch.setattr(synthseg_cli.subprocess, "run", _fail)
    with pytest.raises(RuntimeError, match="mri_synthseg failed"):
        synthseg_cli.segment_volume_via_synthseg(
            torch.rand(4, 16, 16), _geometry(4, 16, 16), workdir=tmp_path
        )


def test_raises_when_no_output_produced(monkeypatch, tmp_path):
    def _noop(argv, **_kw):
        return subprocess.CompletedProcess(argv, 0, "", "")  # rc=0 but writes nothing

    monkeypatch.setattr(synthseg_cli.subprocess, "run", _noop)
    with pytest.raises(RuntimeError, match="no label volume"):
        synthseg_cli.segment_volume_via_synthseg(
            torch.rand(4, 16, 16), _geometry(4, 16, 16), workdir=tmp_path
        )


def test_rejects_non_3d_volume(tmp_path):
    with pytest.raises(ValueError, match=r"\[S, H, W\]"):
        synthseg_cli.segment_volume_via_synthseg(
            torch.rand(1, 4, 16, 16), _geometry(4, 16, 16), workdir=tmp_path
        )


def _fake_synthseg_thresholded():
    """CLI stand-in whose label map tracks content: label 10 where volume > 0.5."""

    def _run(argv, **_kw):
        import nibabel as nib

        i = Path(argv[argv.index("--i") + 1])
        o = Path(argv[argv.index("--o") + 1])
        vol = nib.load(str(i)).get_fdata()
        lab = np.where(vol > 0.5, 10.0, 0.0).astype(np.float32)
        nib.save(nib.Nifti1Image(lab, np.eye(4)), str(o))
        return subprocess.CompletedProcess(argv, 0, "", "")

    return _run


def test_synthseg_dice_trajectories_falls_with_severity(monkeypatch, tmp_path):
    monkeypatch.setattr(synthseg_cli.subprocess, "run", _fake_synthseg_thresholded())

    clean = torch.ones(4, 16, 16)  # all > 0.5 -> label 10 everywhere

    def degrade(vol, family, theta):
        return vol * (1.0 - theta)  # theta->1 erases the label-10 support

    traj = synthseg_cli.synthseg_dice_trajectories(
        clean,
        _geometry(4, 16, 16),
        degrade_fn=degrade,
        families=["gaussian_noise"],
        severities=[0.0, 0.9],
        workdir=tmp_path,
        labels=(10,),
    )
    assert set(traj) == {"gaussian_noise"}
    t = traj["gaussian_noise"]
    assert len(t) == 2
    assert all(0.0 <= v <= 1.0 for v in t)
    assert t[0] == pytest.approx(1.0)  # theta=0: unchanged -> perfect Dice
    assert t[1] < t[0]  # severity erodes the anatomy -> Dice drops (ADR-rankable)
