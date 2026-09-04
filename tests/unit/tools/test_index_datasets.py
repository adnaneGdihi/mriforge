"""Unit tests for spectramr/tools/index_datasets.py.

Regression coverage for the header-parse fix: the ISMRMRD-header block used to
probe a nonexistent ``f.dataset`` attribute, which raised ``AttributeError`` on
every k-space file. That error was caught by the outer ``except`` (formerly
bare, now ``except Exception``) and the whole file was silently skipped — so the
index recorded *no* entries for valid FastMRI-style files. The fix probes the
``xml`` dataset / ``ismrmrd_header`` attribute directly and never touches
``f.dataset``.

These tests build a tiny temp ``.h5`` with both an ``xml`` (ISMRMRD header)
dataset and a ``kspace`` dataset, run the public indexing function over the
directory, and assert it completes WITHOUT raising / swallowing
``AttributeError`` and that it records an entry.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from spectramr.tools.index_datasets import create_index

pytestmark = pytest.mark.unit


ISMRMRD_XML = (
    b"<ismrmrdHeader xmlns='http://www.ismrm.org/ISMRMRD'>"
    b"<measurementInformation>"
    b"<institutionName>TestSite</institutionName>"
    b"</measurementInformation>"
    b"<acquisitionSystemInformation>"
    b"<systemVendor>TestVendor</systemVendor>"
    b"<systemModel>TestModel</systemModel>"
    b"<receiverChannels>4</receiverChannels>"
    b"</acquisitionSystemInformation>"
    b"</ismrmrdHeader>"
)


def _make_h5(path: Path, *, with_xml: bool = True) -> None:
    """Create a tiny FastMRI-style HDF5 file with a kspace + xml dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=np.zeros((2, 4, 4), dtype=np.complex64))
        if with_xml:
            # FastMRI stores the ISMRMRD header in an ``xml`` dataset whose
            # single element is a byte string.
            f.create_dataset("xml", data=np.array([ISMRMRD_XML]))


def test_create_index_does_not_raise_attribute_error_and_records_entry(tmp_path):
    """The header-parse block must not raise/swallow AttributeError, and the
    valid k-space file must end up in the manifest."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_h5(data_root / "sample.h5", with_xml=True)
    out = tmp_path / "index.pkl"

    captured: list[str] = []

    # Capture everything the tool prints so we can prove no file was skipped
    # due to AttributeError (the old bug path printed "Skipping ...: ...").
    with patch(
        "spectramr.tools.index_datasets.print",
        side_effect=lambda *a, **k: captured.append(" ".join(str(x) for x in a)),
    ):
        # Must not raise.
        create_index(
            data_root=str(data_root),
            output_path=str(out),
            pattern="**/*.h5",
            use_relative_paths=True,
        )

    # No file was skipped at all, and certainly not because of AttributeError.
    skipped = [line for line in captured if line.startswith("Skipping")]
    assert skipped == [], f"file(s) unexpectedly skipped: {skipped}"
    assert not any("AttributeError" in line for line in captured), captured

    # The manifest exists and records exactly one entry.
    assert out.exists()
    with open(out, "rb") as f:
        manifest = pickle.load(f)
    assert manifest["version"] == 2
    assert len(manifest["files"]) == 1

    entry = manifest["files"][0]
    assert entry["path"] == "sample.h5"
    assert entry["filename"] == "sample.h5"
    assert tuple(entry["shape"]) == (2, 4, 4)


def test_create_index_parses_ismrmrd_header_metadata(tmp_path):
    """Because the file is no longer silently skipped, the ISMRMRD header in the
    ``xml`` dataset is parsed and surfaced in the entry's metadata."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_h5(data_root / "sample.h5", with_xml=True)
    out = tmp_path / "index.pkl"

    with patch("spectramr.tools.index_datasets.print"):
        create_index(
            data_root=str(data_root),
            output_path=str(out),
            pattern="**/*.h5",
            use_relative_paths=True,
        )

    with open(out, "rb") as f:
        manifest = pickle.load(f)

    metadata = manifest["files"][0]["metadata"]
    # Proof the header-parse block actually executed (the old AttributeError
    # path would have produced an empty / absent entry).
    assert metadata.get("institutionName") == "TestSite"
    assert metadata.get("systemVendor") == "TestVendor"
    assert metadata.get("systemModel") == "TestModel"
    assert metadata.get("receiverChannels") == "4"


def test_create_index_kspace_without_xml_still_indexed(tmp_path):
    """A k-space file lacking an xml header must still be indexed (this is the
    exact case the old ``f.dataset`` probe broke — every kspace file raised)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_h5(data_root / "noxml.h5", with_xml=False)
    out = tmp_path / "index.pkl"

    captured: list[str] = []
    with patch(
        "spectramr.tools.index_datasets.print",
        side_effect=lambda *a, **k: captured.append(" ".join(str(x) for x in a)),
    ):
        create_index(
            data_root=str(data_root),
            output_path=str(out),
            pattern="**/*.h5",
            use_relative_paths=True,
        )

    assert not any("AttributeError" in line for line in captured), captured
    assert [line for line in captured if line.startswith("Skipping")] == []

    with open(out, "rb") as f:
        manifest = pickle.load(f)
    assert len(manifest["files"]) == 1
    # No header -> metadata is an (empty) dict, not a missing key.
    assert manifest["files"][0]["metadata"] == {}
