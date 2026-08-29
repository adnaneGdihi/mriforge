"""Integration: bart/ismrmrd/oracle dispatchers honour ``index_path`` (Mechanism A).

Proves the loader builds its index from the manifest (not the disk glob) when
``index_path`` is set, by pointing ``data_root`` at a NONEXISTENT directory — the
glob path would raise ``FileNotFoundError``; the manifest path ignores
``data_root`` and uses the manifest's embedded one, so success proves the branch.
Also proves the back-compat glob path is unchanged when ``index_path`` is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mriforge.config.schemas.data import BartConfigSchema
from mriforge.data.builders.dataset_instantiator import DatasetInstantiator
from tests.utils.data_config_stub import DataConfigStub


def _write_bart(directory: Path, name: str, arr: np.ndarray):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.hdr").write_text(
        "# Dimensions\n" + " ".join(str(d) for d in arr.shape) + "\n"
    )
    arr.astype(np.complex64).flatten(order="F").tofile(directory / f"{name}.cfl")


def _manifest(path: Path, data_root: Path, rels: list[str], file_type="bart"):
    path.write_text(
        json.dumps(
            {
                "manifest_version": "3.0",
                "dataset_name": "demo",
                "data_root": str(data_root),
                "file_type": file_type,
                "total_records": len(rels),
                "records": [{"relative_path": r, "file_id": r} for r in rels],
                "status": "downloaded",
            }
        )
    )


def test_bart_dispatch_uses_manifest_when_index_path_set(tmp_path: Path):
    root = tmp_path / "raw"
    arr = np.ones((4, 4, 1), dtype=np.complex64)
    _write_bart(root, "scanA", arr)
    _write_bart(root, "scanB", arr)
    mf = tmp_path / "demo.json"
    _manifest(mf, root, ["scanA", "scanB"])

    cfg = DataConfigStub(
        bart=BartConfigSchema(
            enabled=True, sampling="cartesian", bart_dim_map={"readout": 0, "phase": 1, "coil": 2}
        ),
        data_root=str(tmp_path / "does_not_exist"),  # glob path would raise here
        validation_split=0.5,
        index_path=str(mf),
    )

    train, val = DatasetInstantiator._create_bart_dataset(cfg, None, None)

    assert len(train) + len(val) == 2  # both records came from the manifest


def test_bart_dispatch_glob_path_unchanged_without_index_path(tmp_path: Path):
    """Back-compat: no index_path → original data_root+glob index build."""
    root = tmp_path / "raw"
    _write_bart(root, "g1", np.ones((4, 4, 1), dtype=np.complex64))
    _write_bart(root, "g2", np.ones((4, 4, 1), dtype=np.complex64))

    cfg = DataConfigStub(
        bart=BartConfigSchema(
            enabled=True, sampling="cartesian", bart_dim_map={"readout": 0, "phase": 1, "coil": 2}
        ),
        data_root=str(root),
        validation_split=0.5,
        index_path=None,
    )

    train, val = DatasetInstantiator._create_bart_dataset(cfg, None, None)
    assert len(train) + len(val) == 2


def test_bart_dispatch_unpopulated_manifest_raises(tmp_path: Path):
    """An empty-records manifest must fail loud, not silently build nothing."""
    mf = tmp_path / "demo.json"
    _manifest(mf, tmp_path / "raw", [])
    cfg = DataConfigStub(
        bart=BartConfigSchema(enabled=True, bart_dim_map={"readout": 0, "phase": 1, "coil": 2}),
        data_root=str(tmp_path / "x"),
        validation_split=0.5,
        index_path=str(mf),
    )
    with pytest.raises(FileNotFoundError):
        DatasetInstantiator._create_bart_dataset(cfg, None, None)
