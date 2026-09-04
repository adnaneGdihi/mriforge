"""Regression test for the audit-16-F2 fix.

``src/data/builders/torchio_subject_builder.py::_load_tensor`` used to
inline its own ``torch.load`` / ``h5py.File`` / ``nib.load`` dispatch.
It now delegates to ``spectramr.data.io_strategies.load_tensor_from_file``
(the canonical multi-format reader). Pinning the helper here so a
future refactor can't silently re-inline the dispatch (CLAUDE.md
pitfall #11: data-loading SSOT under ``src/data/``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.data.io_strategies import load_tensor_from_file


def test_load_pt_tensor(tmp_path) -> None:
    p = tmp_path / "x.pt"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(expected, p)
    out = load_tensor_from_file(p)
    assert torch.equal(out, expected)


def test_load_npy_tensor(tmp_path) -> None:
    p = tmp_path / "x.npy"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(p, arr)
    out = load_tensor_from_file(p)
    assert torch.equal(out, torch.from_numpy(arr))


def test_load_h5_tensor(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "x.h5"
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=arr)
    out = load_tensor_from_file(p)
    assert torch.equal(out, torch.from_numpy(arr))


def test_load_h5_respects_key_preference(tmp_path) -> None:
    """Review 2026-07-01: ``h5_keys`` selects the first PRESENT key so inference
    (which prefers ``kspace`` over ``reconstruction_*``) can route through this
    SSOT loader instead of its own inline h5py reader."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "multi.h5"
    ksp = np.arange(6, dtype=np.float32).reshape(2, 3)
    rec = np.zeros((2, 3), dtype=np.float32)
    with h5py.File(p, "w") as f:
        # Insert reconstruction_rss FIRST to prove precedence beats insertion order.
        f.create_dataset("reconstruction_rss", data=rec)
        f.create_dataset("kspace", data=ksp)
    out = load_tensor_from_file(p, h5_keys=("kspace", "reconstruction_rss", "data"))
    assert torch.equal(out, torch.from_numpy(ksp))


def test_load_h5_falls_back_to_first_key_when_no_preference_matches(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "one.h5"
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    with h5py.File(p, "w") as f:
        f.create_dataset("only", data=arr)
    out = load_tensor_from_file(p, h5_keys=("kspace", "data"))  # none present
    assert torch.equal(out, torch.from_numpy(arr))


def test_load_hdf5_extension_supported(tmp_path) -> None:
    """``.hdf5`` must load identically to ``.h5`` (inference used ``.hdf5``)."""
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "x.hdf5"
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    with h5py.File(p, "w") as f:
        f.create_dataset("data", data=arr)
    out = load_tensor_from_file(p)
    assert torch.equal(out, torch.from_numpy(arr))


def test_unsupported_format_raises(tmp_path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("not a tensor")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_tensor_from_file(p)


def test_load_nii_gz_tensor(tmp_path) -> None:
    """IOM-001: a real ``.nii.gz`` path normalises to ``.nii.gz`` and loads."""
    nib = pytest.importorskip("nibabel")
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    p = tmp_path / "vol.nii.gz"
    nib.save(img, str(p))
    out = load_tensor_from_file(p)
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.from_numpy(arr).float())


def test_unsupported_gz_raises(tmp_path) -> None:
    """IOM-001: an unrelated ``.gz`` (e.g. ``.tar.gz``) must NOT route to nibabel."""
    p = tmp_path / "x.tar.gz"
    p.write_bytes(b"\x1f\x8b\x08\x00not a nifti")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_tensor_from_file(p)


def test_subject_builder_load_tensor_delegates_to_io_strategies() -> None:
    """Source-level guard against the dispatch logic being re-inlined."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "spectramr"
        / "data"
        / "builders"
        / "torchio_subject_builder.py"
    ).read_text()

    # Locate the _load_tensor body — between its `def` and the next
    # top-level `def` or `class`. The body should NOT contain inline
    # h5py/nib/torch.load dispatch.
    start = src.index("def _load_tensor(")
    end = src.find("\n    def ", start + 1)
    if end == -1:
        end = src.find("\nclass ", start + 1)
    body = src[start:end]
    for needle in ("h5py.File(", "nib.load(", "torch.load("):
        assert needle not in body, (
            f"torchio_subject_builder._load_tensor re-inlined `{needle}`. "
            "Route through spectramr.data.io_strategies.load_tensor_from_file — "
            "see audit 16 F2."
        )
