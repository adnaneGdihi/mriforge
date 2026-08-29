"""Tiny synthetic fixtures for the dataset-EDA unit tests (no real-data dependency)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_h5_kspace(path: Path, coils: int = 4, h: int = 32, w: int = 32, slices: int = 3) -> None:
    import h5py

    rng = np.random.default_rng(0)
    img = rng.standard_normal((slices, coils, h, w)) + 1j * rng.standard_normal((slices, coils, h, w))
    ksp = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1)).astype(np.complex64)
    rss = np.sqrt((np.abs(img) ** 2).sum(axis=1)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=ksp)
        f.create_dataset("reconstruction_rss", data=rss)


@pytest.fixture
def present_kspace_manifest(tmp_path: Path) -> Path:
    """A populated local-style manifest pointing at one tiny multicoil h5."""
    root = tmp_path / "db" / "ks"
    root.mkdir(parents=True)
    _write_h5_kspace(root / "rec0.h5")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    mpath = manifests / "tiny_kspace.json"
    mpath.write_text(
        json.dumps(
            {
                "dataset_name": "tiny_kspace",
                "data_root": str(root),
                "file_type": "h5",
                "records": [
                    {"relative_path": "rec0.h5", "filename": "rec0.h5", "shape": [3, 4, 32, 32]}
                ],
            }
        )
    )
    return mpath


@pytest.fixture
def absent_external_manifest(tmp_path: Path) -> Path:
    """A not_downloaded external-style manifest (empty records, rich source)."""
    manifests = tmp_path / "manifests" / "external"
    manifests.mkdir(parents=True)
    mpath = manifests / "calgary_campinas.json"
    mpath.write_text(
        json.dumps(
            {
                "dataset_name": "calgary_campinas",
                "data_root": "databases/external/calgary_campinas/raw",
                "file_type": "h5",
                "records": [],
                "status": "not_downloaded",
                "local_manifest": None,
                "source": {
                    "name": "Calgary-Campinas CC-359",
                    "role": "RAW",
                    "raw_kspace": True,
                    "field_T": [1.5, 3.0],
                    "anatomy": "brain",
                    "provides": "raw 3D GRE",
                    "links": ["https://www.ccdataset.com/download"],
                },
            }
        )
    )
    return mpath
