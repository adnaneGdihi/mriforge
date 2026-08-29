"""IOM-004 regression test for manifest-roles format dispatch.

``IndexBuilder.load_from_manifest_roles`` dispatches on each manifest
record's ``format`` field (``h5`` / ``nifti`` / ``nii`` / ``pt`` / ``png``).
The dispatch block previously had no ``else`` branch, so an unknown
format value (e.g. ``zarr``) silently contributed nothing to the subject
— an NN#3 / pitfall #9 silent fallback. The fix raises ``ValueError`` on
any unrecognised format. This test pins that behaviour.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from mriforge.config.schemas.data import DataSplitConfigSchema
from mriforge.data.metadata.index_builder import IndexBuilder
from tests.utils.data_config_stub import DataConfigStub


def _write_manifest(path, samples) -> None:
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def test_load_from_manifest_roles_unknown_format_raises(tmp_path) -> None:
    """An unknown ``format`` value must raise ValueError, not silently skip."""
    data_file = tmp_path / "sample.zarr"
    data_file.write_bytes(b"not real zarr")

    manifest_path = tmp_path / "input_manifest.pkl"
    _write_manifest(
        manifest_path,
        [
            {
                "filename": "sample.zarr",
                "path": str(data_file),
                "format": "zarr",  # unsupported
            }
        ],
    )

    roles = SimpleNamespace(
        inputs=[{"manifest": str(manifest_path), "key": "input"}],
        targets=[],
        auxiliary=[],
    )
    config = DataConfigStub(
        manifest_roles=roles,
        data_root=None,
        # validation_fraction=0 -> explicit train-only. A single-file manifest
        # cannot form a non-overlapping split, and the loader raises rather than
        # reusing the file for both (that would leak val into train).
        split=DataSplitConfigSchema(validation_fraction=0),
    )

    with pytest.raises(ValueError, match="unknown format 'zarr'"):
        IndexBuilder.load_from_manifest_roles(config, None, None)


def test_load_from_manifest_roles_nifti_format_does_not_raise(tmp_path) -> None:
    """A supported format ('nifti') must NOT trigger the unknown-format raise."""
    nib = pytest.importorskip("nibabel")
    import numpy as np

    img = nib.Nifti1Image(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2), affine=np.eye(4)
    )
    data_file = tmp_path / "sample.nii.gz"
    nib.save(img, str(data_file))

    manifest_path = tmp_path / "input_manifest.pkl"
    _write_manifest(
        manifest_path,
        [
            {
                "filename": "sample.nii.gz",
                "path": str(data_file),
                "format": "nifti",
            }
        ],
    )

    roles = SimpleNamespace(
        inputs=[{"manifest": str(manifest_path), "key": "input"}],
        targets=[],
        auxiliary=[],
    )
    config = DataConfigStub(
        manifest_roles=roles,
        data_root=None,
        # validation_fraction=0 -> explicit train-only. A single-file manifest
        # cannot form a non-overlapping split, and the loader raises rather than
        # reusing the file for both (that would leak val into train).
        split=DataSplitConfigSchema(validation_fraction=0),
    )

    # A recognised format must not raise the unknown-format ValueError; it
    # builds a valid TorchIO subject and returns (train, val) lists.
    train_subjects, val_subjects = IndexBuilder.load_from_manifest_roles(
        config, None, None
    )
    assert isinstance(train_subjects, list)
    assert isinstance(val_subjects, list)
    assert len(train_subjects) >= 1
