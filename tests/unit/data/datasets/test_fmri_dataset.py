"""Unit tests for FMRIVolumeDataset load-failure handling.

A file whose I/O fails must NOT be silently replaced by an all-zeros volume —
that trains the model on fabricated data while looking healthy (pitfall #9/#16,
the same zero-fill facade removed from m4raw_dataset). ``__getitem__`` raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from spectramr.data.datasets.fmri_dataset import FMRIVolumeDataset  # noqa: E402


def _write_valid_npy(path: Path) -> None:
    np.save(path, np.random.rand(4, 8, 8, 1).astype("float32"))


def test_valid_volume_loads(tmp_path: Path) -> None:
    _write_valid_npy(tmp_path / "vol_a.npy")
    ds = FMRIVolumeDataset(tmp_path)
    assert len(ds) == 1
    sample = ds[0]
    assert "image" in sample


def test_corrupt_volume_raises_not_zero_fill(tmp_path: Path) -> None:
    """A ``.npy`` that fails to load must raise, not yield a zeros placeholder."""
    bad = tmp_path / "corrupt.npy"
    bad.write_bytes(b"not a valid npy header")
    ds = FMRIVolumeDataset(tmp_path)
    assert len(ds) == 1
    with pytest.raises(RuntimeError, match="corrupt|failed|load"):
        _ = ds[0]
