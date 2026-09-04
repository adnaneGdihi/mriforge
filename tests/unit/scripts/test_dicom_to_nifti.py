"""Tests for ``scripts/preprocessing/dicom_to_nifti.py`` DICOM *discovery*.

The manifest's ``--populate`` routing tells the operator to convert a dataset of
raw DICOM to NIfTI with this script (raw DICOM is not loader-readable). That
routing fires precisely when the ingestion detector
(``scripts/data/_dataset_file_types.py::magic_format``) identifies DICOM **by its
``DICM`` Part-10 preamble** — which only happens when the files carry no ``.dcm``
extension (a manual download saved them under an accession id or a numeric PACS
name like ``IM-0001.06666``).

The converter previously discovered DICOM *only* by ``*.dcm`` glob, so on exactly
those extensionless datasets it found zero files: the routed "convert first"
command was a silent no-op (pitfall #16, facade). These tests pin that the
converter discovers DICOM the same way the detector does — by extension OR magic
byte — so detector and converter cannot disagree.

Discovery is tested against a synthetic byte tree (a 128-byte preamble + ``DICM``
is enough for magic identification); the heavy pydicom pixel decode is not
exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.repo_scripts import load_script_module

# The script imports `pydicom` and `nibabel`, neither declared in pyproject and both
# absent from a clean `[dev]` env. The load below EXECUTES it at module scope, so a
# missing dep aborts collection for all of tests/unit (#804, CI-only variant).
#
# The SCRIPT going missing is the same hazard reached by a different absence, and it
# was unguarded: `scripts/preprocessing/` is not in the public allowlist, so in the
# export this raised `FileNotFoundError` during collection. Both deps happen to be
# installed transitively, so the importorskips above do NOT mask it. Hence the
# helper, which skips in the export and fails loudly in a private checkout.
pytest.importorskip("pydicom")
pytest.importorskip("nibabel")

conv = load_script_module("scripts/preprocessing/dicom_to_nifti.py", "dicom_to_nifti")

# A minimal DICOM Part-10 stream: 128-byte preamble then the ``DICM`` magic.
_DICOM = b"\x00" * 128 + b"DICM" + b"\x00" * 64


def test_find_dicom_files_by_extension(tmp_path: Path):
    """A ``.dcm`` file is discovered (the pre-existing extension path is kept)."""
    f = tmp_path / "slice.dcm"
    f.write_bytes(_DICOM)
    found = conv.find_dicom_files(tmp_path)
    assert found == [f]


def test_find_dicom_files_extensionless_by_magic(tmp_path: Path):
    """An extensionless file with the ``DICM`` preamble is discovered by magic.

    This is the regression: ``multiwave_005t`` / numeric-ext PACS DICOM carry no
    ``.dcm`` suffix, so an extension-only glob misses them and the routed convert
    command silently converts nothing.
    """
    f = tmp_path / "1.2.840.113619.2.1"  # numeric PACS name, no extension
    f.write_bytes(_DICOM)
    found = conv.find_dicom_files(tmp_path)
    assert found == [f]


def test_find_dicom_files_numeric_extension_by_magic(tmp_path: Path):
    """A ``.06666``-style numeric extension is discovered by magic, not suffix."""
    f = tmp_path / "2412.06666"
    f.write_bytes(_DICOM)
    assert conv.find_dicom_files(tmp_path) == [f]


def test_find_dicom_files_ignores_non_dicom(tmp_path: Path):
    """A non-DICOM file (no ``.dcm`` ext, no ``DICM`` magic) is NOT discovered."""
    (tmp_path / "readme").write_bytes(b"<!DOCTYPE html>\n")
    (tmp_path / "vol.nii.gz").write_bytes(b"\x1f\x8b\x08\x00not-a-dicom")
    assert conv.find_dicom_files(tmp_path) == []


def test_find_dicom_files_recursive(tmp_path: Path):
    """Discovery recurses into subdirectories by default."""
    sub = tmp_path / "series01"
    sub.mkdir()
    f = sub / "instance"
    f.write_bytes(_DICOM)
    assert conv.find_dicom_files(tmp_path) == [f]


def test_find_dicom_files_non_recursive_skips_subdirs(tmp_path: Path):
    """``recursive=False`` only scans the top level."""
    top = tmp_path / "top.dcm"
    top.write_bytes(_DICOM)
    sub = tmp_path / "series01"
    sub.mkdir()
    (sub / "nested").write_bytes(_DICOM)
    assert conv.find_dicom_files(tmp_path, recursive=False) == [top]


def test_find_dicom_files_reuses_detector_magic(tmp_path: Path):
    """The converter identifies DICOM via the SAME ``magic_format`` the ingestion
    detector uses — one signature, two consumers (no second source of truth)."""
    f = tmp_path / "no_ext"
    f.write_bytes(_DICOM)
    # The shared SSOT identifies this byte pattern as dicom...
    assert conv.magic_format(f) == "dicom"
    # ...and the converter's discovery agrees.
    assert conv.find_dicom_files(tmp_path) == [f]


def test_find_dicom_files_siemens_ima_by_extension(tmp_path: Path):
    """A Siemens ``.IMA`` file is discovered by extension even WITHOUT a ``DICM``
    preamble. Modern Skyra IMAs are DICOM, but the extension path must not depend on
    the magic byte (older Siemens IMA carry no Part-10 preamble) — else the routed
    convert command silently converts nothing on a phantom-DICOM dataset."""
    f = tmp_path / "STJUDE_Siemens_SKYRA_3T_08132015.IMA"
    f.write_bytes(b"\x08\x00" * 64)  # no 128-byte preamble + DICM
    # magic alone would NOT identify this byte pattern as dicom...
    assert conv.magic_format(f) != "dicom"
    # ...but the converter discovers it by the ``.IMA`` (Siemens DICOM) extension.
    assert conv.find_dicom_files(tmp_path) == [f]
