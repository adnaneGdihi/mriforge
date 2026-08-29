"""Tests for :class:`mriforge.data.datasets.inference_dataset.InferenceDataset`.

Focus: the dataset serves EVERY slice of a 3-D volume as its own inference
sample, rather than silently collapsing each volume to its central slice. An
explicit ``slice_index`` still pins a single slice per file (legacy behaviour).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mriforge.data.datasets.inference_dataset import InferenceDataset


def _write_nifti(path: Path, array: np.ndarray) -> None:
    import nibabel as nib

    nib.save(nib.Nifti1Image(array, affine=np.eye(4)), str(path))


def _make_volume(path: Path, depth: int = 3, hw: int = 8) -> np.ndarray:
    # Each slice is a distinct constant-ramp so slices are trivially separable.
    vol = np.stack(
        [
            np.full((hw, hw), float(100 * (s + 1)), dtype=np.float32)
            for s in range(depth)
        ]
    )
    vol[:, 0, 0] = 0.0  # a real span so normalisation is not degenerate
    _write_nifti(path, vol)
    return vol


def test_serves_all_slices_of_a_3d_nifti(tmp_path: Path) -> None:
    """A depth-D volume yields D inference samples, one per slice."""
    path = tmp_path / "sub-01.nii.gz"
    _make_volume(path, depth=3, hw=8)

    ds = InferenceDataset([path], target_hw=(8, 8))
    assert len(ds) == 3


def test_each_slice_item_has_a_unique_key(tmp_path: Path) -> None:
    """Per-slice items carry distinct keys so outputs never overwrite."""
    path = tmp_path / "sub-01.nii.gz"
    _make_volume(path, depth=3, hw=8)

    ds = InferenceDataset([path], target_hw=(8, 8))
    keys = [ds[i][0] for i in range(len(ds))]
    assert len(set(keys)) == len(keys) == 3


def test_slices_are_distinct_tensors(tmp_path: Path) -> None:
    """Different slices produce different input tensors (not one repeated slice)."""
    path = tmp_path / "sub-01.nii.gz"
    _make_volume(path, depth=3, hw=8)

    ds = InferenceDataset([path], target_hw=(8, 8))
    t0 = ds[0][1]
    t1 = ds[1][1]
    assert not np.allclose(t0.numpy(), t1.numpy())


def test_explicit_slice_index_yields_one_item_per_file(tmp_path: Path) -> None:
    """A pinned slice_index keeps the legacy one-sample-per-file behaviour."""
    path = tmp_path / "sub-01.nii.gz"
    _make_volume(path, depth=3, hw=8)

    ds = InferenceDataset([path], slice_index=1, target_hw=(8, 8))
    assert len(ds) == 1


def test_2d_input_yields_single_item(tmp_path: Path) -> None:
    """A 2-D input contributes exactly one sample."""
    arr = (np.random.default_rng(0).random((8, 8)).astype(np.float32)) * 1000.0
    path = tmp_path / "slice2d.nii.gz"
    _write_nifti(path, arr)

    ds = InferenceDataset([path], target_hw=(8, 8))
    assert len(ds) == 1


def test_all_slices_across_multiple_files(tmp_path: Path) -> None:
    """The index spans every slice of every file (2 files x 3 + 2 slices)."""
    p1 = tmp_path / "sub-01.nii.gz"
    p2 = tmp_path / "sub-02.nii.gz"
    _make_volume(p1, depth=3, hw=8)
    _make_volume(p2, depth=2, hw=8)

    ds = InferenceDataset([p1, p2], target_hw=(8, 8))
    assert len(ds) == 5


class TestSliceCountProbeAvoidsDecoding:
    """#393 (audit D4). Indexing probes EVERY input file at construction.

    A probe that decodes is paid once per file before a single sample is
    served, and then largely paid AGAIN at serve time: the raw LRU holds only
    ``cache_size`` volumes, so on any corpus larger than the cache the probe's
    decode is evicted before it is used.
    """

    def test_npy_shape_comes_from_the_header(self, tmp_path) -> None:
        import numpy as np

        from mriforge.data.datasets.inference_dataset import InferenceDataset

        path = tmp_path / "vol.npy"
        np.save(path, np.zeros((7, 32, 32), dtype=np.float32))
        assert InferenceDataset._probe_shape_no_decode(path) == (7, 32, 32)

    def test_h5_shape_comes_from_the_header(self, tmp_path) -> None:
        h5py = pytest.importorskip("h5py")

        from mriforge.data.datasets.inference_dataset import InferenceDataset

        path = tmp_path / "vol.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", shape=(5, 4, 16, 16, 2), dtype="f4")
        assert InferenceDataset._probe_shape_no_decode(path) == (5, 4, 16, 16, 2)

    def test_the_npy_probe_does_not_read_the_data_block(self, tmp_path) -> None:
        """The point of the change, asserted rather than assumed.

        A truncated file keeps a valid header but has no array behind it, so a
        probe that returns the right shape provably never touched the data —
        ``np.load`` on the same file raises.
        """
        import numpy as np

        from mriforge.data.datasets.inference_dataset import InferenceDataset

        full = tmp_path / "full.npy"
        np.save(full, np.zeros((9, 64, 64), dtype=np.float32))
        truncated = tmp_path / "truncated.npy"
        truncated.write_bytes(full.read_bytes()[:256])

        assert InferenceDataset._probe_shape_no_decode(truncated) == (9, 64, 64)
        with pytest.raises(ValueError):
            np.load(truncated)

    def test_an_unprobeable_format_returns_none_not_a_guess(self, tmp_path) -> None:
        """``None`` routes the caller to a real decode. A guessed count would
        serve the wrong number of slices, silently."""
        from mriforge.data.datasets.inference_dataset import InferenceDataset

        path = tmp_path / "image.png"
        path.write_bytes(b"not really a png")
        assert InferenceDataset._probe_shape_no_decode(path) is None

    def test_a_corrupt_header_is_not_fatal(self, tmp_path) -> None:
        """A probe must never be the thing that kills the run."""
        from mriforge.data.datasets.inference_dataset import InferenceDataset

        path = tmp_path / "bad.npy"
        path.write_bytes(b"\x00" * 32)
        assert InferenceDataset._probe_shape_no_decode(path) is None
