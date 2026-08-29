"""Shared fixtures for preprocessing pipeline tests.

Builds synthetic filesystem layouts that mirror real dataset structures:
- BIDS-compliant paired ULF/HF brain datasets
- Flat NIfTI datasets
- FastMRI-style k-space HDF5 datasets
- M4Raw multicoil datasets

All fixtures use tmp_path so tests remain hermetic.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — same as the production entry point
# ---------------------------------------------------------------------------
_PREPROC_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "preprocessing"
_REPO_ROOT = _PREPROC_ROOT.parent.parent

if str(_PREPROC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PREPROC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# NIfTI synthesis helper (minimal valid .nii.gz — no nibabel needed)
# ---------------------------------------------------------------------------
def create_synthetic_nifti(
    path: Path,
    shape: tuple[int, int, int] = (32, 32, 8),
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 5.0),
    dtype: np.dtype = np.float32,
    seed: int = 42,
) -> Path:
    """Create a minimal NIfTI-1 file that nibabel can load.

    Uses only numpy and gzip — no nibabel dependency.
    """
    import gzip

    rng = np.random.RandomState(seed)
    data = rng.randn(*shape).astype(dtype)

    # NIfTI-1 header (348 bytes)
    header = bytearray(348)
    # sizeof_hdr
    struct.pack_into("<i", header, 0, 348)
    # dim
    struct.pack_into("<h", header, 40, 3)
    struct.pack_into("<h", header, 42, shape[0])
    struct.pack_into("<h", header, 44, shape[1])
    struct.pack_into("<h", header, 46, shape[2])
    # datatype = float32 (16)
    struct.pack_into("<h", header, 70, 16)
    # bitpix = 32
    struct.pack_into("<h", header, 72, 32)
    # pixdim
    struct.pack_into("<f", header, 76, 1.0)  # qfac
    struct.pack_into("<f", header, 80, voxel_size[0])
    struct.pack_into("<f", header, 84, voxel_size[1])
    struct.pack_into("<f", header, 88, voxel_size[2])
    # vox_offset
    struct.pack_into("<f", header, 108, 352.0)
    # magic "n+1\0"
    header[344:348] = b"n+1\0"

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = bytes(header) + b"\x00" * 4 + data.tobytes()
    with gzip.open(str(path), "wb") as f:
        f.write(raw)
    return path


def create_synthetic_h5(
    path: Path,
    shape: tuple[int, ...] = (4, 16, 256, 256),
    seed: int = 42,
) -> Path:
    """Create a minimal HDF5 file with a 'kspace' dataset."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.RandomState(seed)
    kspace = (rng.randn(*shape) + 1j * rng.randn(*shape)).astype(np.complex64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=kspace)
        f.attrs["acquisition"] = "AXFlair"
    return path


# ---------------------------------------------------------------------------
# Dataset layout builders
# ---------------------------------------------------------------------------
@pytest.fixture
def bids_ulf_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a BIDS-compliant paired ULF/HF dataset tree.

    Layout::

        {tmp}/ulf_data/sub-0001/anat/sub-0001_T1w.nii.gz
        {tmp}/ulf_data/sub-0001/anat/sub-0001_T2w.nii.gz
        {tmp}/ulf_data/sub-0002/anat/sub-0002_T1w.nii.gz
        {tmp}/hf_data/sub-0001/anat/sub-0001_T1w.nii.gz
        {tmp}/hf_data/sub-0001/anat/sub-0001_T2w.nii.gz
        {tmp}/hf_data/sub-0002/anat/sub-0002_T1w.nii.gz
    """
    ulf_root = tmp_path / "ulf_data"
    hf_root = tmp_path / "hf_data"

    for sub_id in ("sub-0001", "sub-0002"):
        for contrast in ("T1w", "T2w"):
            # ULF side
            ulf_path = ulf_root / sub_id / "anat" / f"{sub_id}_{contrast}.nii.gz"
            create_synthetic_nifti(ulf_path, seed=hash(f"ulf_{sub_id}_{contrast}") % 1000)

            # HF side
            hf_path = hf_root / sub_id / "anat" / f"{sub_id}_{contrast}.nii.gz"
            create_synthetic_nifti(hf_path, seed=hash(f"hf_{sub_id}_{contrast}") % 1000)

    # sub-0003 only has ULF (no HF pair)
    ulf_path = ulf_root / "sub-0003" / "anat" / "sub-0003_T1w.nii.gz"
    create_synthetic_nifti(ulf_path, seed=99)

    return {
        "root": tmp_path,
        "ulf_root": ulf_root,
        "hf_root": hf_root,
        "contrasts": ["T1w", "T2w"],
        "paired_subjects": ["sub-0001", "sub-0002"],
        "unpaired_subjects": ["sub-0003"],
    }


@pytest.fixture
def flat_nifti_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a flat NIfTI dataset (no BIDS, volumes in subject root).

    Layout::

        {tmp}/flat_data/sub-001/T1w.nii.gz
        {tmp}/flat_data/sub-001/T2w.nii.gz
        {tmp}/flat_data/sub-002/T1w.nii.gz
    """
    root = tmp_path / "flat_data"
    for sub_id in ("sub-001", "sub-002"):
        for contrast in ("T1w",):
            p = root / sub_id / f"{contrast}.nii.gz"
            create_synthetic_nifti(p, seed=hash(f"{sub_id}_{contrast}") % 1000)

    p = root / "sub-001" / "T2w.nii.gz"
    create_synthetic_nifti(p, seed=77)

    return {"root": root, "contrasts": ["T1w", "T2w"]}


@pytest.fixture
def bids_session_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a BIDS dataset with sessions.

    Layout::

        {tmp}/ses_data/sub-01/ses-baseline/anat/sub-01_ses-baseline_T1w.nii.gz
        {tmp}/ses_data/sub-01/ses-followup/anat/sub-01_ses-followup_T1w.nii.gz
    """
    root = tmp_path / "ses_data"
    for session in ("baseline", "followup"):
        p = root / "sub-01" / f"ses-{session}" / "anat" / f"sub-01_ses-{session}_T1w.nii.gz"
        create_synthetic_nifti(p, seed=hash(session) % 1000)
    return {"root": root}


@pytest.fixture
def fastmri_kspace_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a synthetic FastMRI-style k-space dataset.

    Layout::

        {tmp}/fastmri_data/file_brain_AXT1_200_001.h5
        {tmp}/fastmri_data/file_brain_AXT1_200_002.h5
        {tmp}/fastmri_data/file_brain_AXFLAIR_200_003.h5
    """
    root = tmp_path / "fastmri_data"
    h5_files = []
    for name in ("file_brain_AXT1_200_001", "file_brain_AXT1_200_002", "file_brain_AXFLAIR_200_003"):
        p = create_synthetic_h5(
            root / f"{name}.h5",
            shape=(16, 8, 128, 128),
            seed=hash(name) % 1000,
        )
        h5_files.append(p)
    return {"root": root, "h5_files": h5_files}


@pytest.fixture
def m4raw_kspace_dataset(tmp_path: Path) -> dict[str, Any]:
    """Build a synthetic M4Raw-style k-space dataset.

    Layout::

        {tmp}/m4raw_data/sub001_T1.h5
        {tmp}/m4raw_data/sub001_T2.h5
    """
    root = tmp_path / "m4raw_data"
    h5_files = []
    for name in ("sub001_T1", "sub001_T2"):
        p = create_synthetic_h5(
            root / f"{name}.h5",
            shape=(4, 18, 256, 256),
            seed=hash(name) % 1000,
        )
        h5_files.append(p)
    return {"root": root, "h5_files": h5_files}


@pytest.fixture
def mock_args(tmp_path: Path) -> argparse.Namespace:
    """Minimal argparse.Namespace for TaskContext."""
    return argparse.Namespace(
        dry_run=False,
        overwrite=True,
        registration_tool="antsxpy",
        work_dir=str(tmp_path / "work"),
        bias_correct=False,
        output_dir=str(tmp_path / "output"),
        dataset_root=str(tmp_path),
        contrasts=None,
        max_workers=1,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_dry_args(tmp_path: Path) -> argparse.Namespace:
    """Minimal argparse.Namespace with dry_run enabled."""
    return argparse.Namespace(
        dry_run=True,
        overwrite=True,
        registration_tool="antsxpy",
        work_dir=str(tmp_path / "work"),
        bias_correct=False,
        output_dir=str(tmp_path / "output"),
        dataset_root=str(tmp_path),
        contrasts=None,
        max_workers=1,
        log_level="DEBUG",
    )
