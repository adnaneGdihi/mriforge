"""Phase 4b-extra — BundleMultiEcho transform."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchio as tio

from mriforge.data.transforms.multi_echo import (
    BundleMultiEcho,
    parse_echo_index_from_filename,
)


# ── Filename parsing ────────────────────────────────────────────────────────


def test_parse_echo_index_bids_underscored() -> None:
    assert parse_echo_index_from_filename("sub-001_echo-3_T2w.nii.gz") == 3


def test_parse_echo_index_double_digit() -> None:
    assert parse_echo_index_from_filename("sub-001_echo-12_PD.nii.gz") == 12


def test_parse_echo_index_returns_none_when_absent() -> None:
    assert parse_echo_index_from_filename("sub-001_T1w.nii.gz") is None


def test_parse_echo_index_underscore_variant() -> None:
    """Some vendors use 'echo_3' instead of 'echo-3'."""
    assert parse_echo_index_from_filename("scan_echo_5.nii.gz") == 5


# ── Validation when input is already multi-channel ──────────────────────────


def test_bundle_passes_when_te_list_matches_channels() -> None:
    """TE list length == channel count ⇒ no-op happy path."""
    data = torch.zeros(3, 4, 4, 2)  # 3 echoes
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0, 20.0]},
    )
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 3


def test_bundle_raises_on_te_length_mismatch() -> None:
    """3 channels but TE list of 5 ⇒ raise (silent mismatch is dangerous)."""
    data = torch.zeros(3, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0, 20.0, 40.0, 80.0]},
    )
    with pytest.raises(ValueError, match="channels"):
        BundleMultiEcho()(subj)


def test_bundle_ignores_te_when_metadata_is_scalar() -> None:
    """Single-echo scan ⇒ TE is float, not list. No validation triggered."""
    data = torch.zeros(1, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": 5.0},
    )
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 1


def test_bundle_skips_validation_when_disabled() -> None:
    """validate_metadata=False ⇒ mismatch tolerated."""
    data = torch.zeros(3, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0]},  # mismatch
    )
    out = BundleMultiEcho(validate_metadata=False)(subj)
    assert out["n_echoes"] == 3


# ── Stack from sibling paths ────────────────────────────────────────────────


def _try_import_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        return None


def test_bundle_loads_and_stacks_sibling_paths(tmp_path: Path) -> None:
    nib = _try_import_nibabel()
    if nib is None:
        pytest.skip("nibabel not installed")
    import numpy as np

    paths: list[str] = []
    for echo_idx in range(3):
        arr = np.full((4, 4, 2), float(echo_idx + 1), dtype=np.float32)
        p = tmp_path / f"sub-001_echo-{echo_idx + 1}_T2w.nii.gz"
        nib.save(  # type: ignore[no-untyped-call]
            nib.Nifti1Image(arr, affine=np.eye(4)), str(p)
        )
        paths.append(str(p))

    # Subject starts with a placeholder single-channel input + a list of
    # sibling paths in metadata.
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)),
        sibling_echo_paths=paths,
    )
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 3
    stacked = out["input"].data
    assert stacked.shape == (3, 4, 4, 2)
    # Each channel should have the constant value from its echo index.
    assert stacked[0, 0, 0, 0] == 1.0
    assert stacked[1, 0, 0, 0] == 2.0
    assert stacked[2, 0, 0, 0] == 3.0
