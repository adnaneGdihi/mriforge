"""Unit tests for multi_echo transforms.

Covers:
  - parse_echo_index_from_filename: BIDS naming patterns (echo-N, echo_N),
    two-digit echo index, missing returns None.
  - BundleMultiEcho:
    - canary on a multi-channel subject with matching metadata,
    - n_echoes recorded on subject,
    - TE list mismatch raises ValueError (silent mismatch corrupts T2 fitting),
    - scalar TE (single-echo) does not trigger list validation,
    - validate_metadata=False skips the check,
    - subject without 'input' key is passed through unchanged.
"""
from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.multi_echo import BundleMultiEcho, parse_echo_index_from_filename


# ── parse_echo_index_from_filename ────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("sub-001_echo-1_T2w.nii.gz", 1),
        ("sub-001_echo-12_PD.nii.gz", 12),
        ("scan_echo_5.nii.gz", 5),           # underscore variant
        ("sub-001_T1w.nii.gz", None),         # no echo token
        ("sub-001_echo-3_mag.nii", 3),        # .nii (not .nii.gz)
    ],
    ids=["bids-1", "bids-12", "underscore-5", "no-echo", "nii-3"],
)
def test_parse_echo_index_from_filename(
    filename: str, expected: int | None
) -> None:
    assert parse_echo_index_from_filename(filename) == expected


# ── BundleMultiEcho: canary ───────────────────────────────────────────────────


def test_canary_multi_channel_with_matching_te() -> None:
    """3-echo subject with TE list of 3 → n_echoes=3, no exception."""
    data = torch.zeros(3, 4, 4, 2)  # 3 channels = 3 echoes
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0, 20.0]},
    )
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 3


# ── n_echoes always recorded ──────────────────────────────────────────────────


def test_n_echoes_recorded_even_without_metadata() -> None:
    data = torch.zeros(2, 4, 4, 2)
    subj = tio.Subject(input=tio.ScalarImage(tensor=data))
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 2


def test_single_echo_records_n_echoes_one() -> None:
    data = torch.zeros(1, 4, 4, 2)
    subj = tio.Subject(input=tio.ScalarImage(tensor=data))
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 1


# ── TE list mismatch raises (not silent) ─────────────────────────────────────


def test_te_list_mismatch_raises_value_error() -> None:
    """3 channels but TE list of 5 → ValueError (pitfall #9)."""
    data = torch.zeros(3, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0, 20.0, 40.0, 80.0]},
    )
    with pytest.raises(ValueError, match="channels"):
        BundleMultiEcho()(subj)


def test_te_list_shorter_than_channels_raises() -> None:
    """TE list too short also raises."""
    data = torch.zeros(4, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0]},
    )
    with pytest.raises(ValueError, match="channels"):
        BundleMultiEcho()(subj)


# ── Scalar TE (single-echo scan) ─────────────────────────────────────────────


def test_scalar_te_does_not_trigger_validation() -> None:
    """Single-echo scan: TE is float, not list — no validation needed."""
    data = torch.zeros(1, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": 5.0},
    )
    out = BundleMultiEcho()(subj)
    assert out["n_echoes"] == 1


# ── validate_metadata=False skips the check ──────────────────────────────────


def test_validate_metadata_false_tolerates_mismatch() -> None:
    data = torch.zeros(3, 4, 4, 2)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=data),
        acquisition_metadata={"TE": [5.0, 10.0]},  # mismatch: 3 vs 2
    )
    out = BundleMultiEcho(validate_metadata=False)(subj)
    assert out["n_echoes"] == 3  # transform ran without raising


# ── No 'input' key ───────────────────────────────────────────────────────────


def test_subject_without_input_key_passes_through() -> None:
    """Subject with only a 'target' image → transform is a no-op (no crash)."""
    subj = tio.Subject(target=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 2)))
    out = BundleMultiEcho()(subj)
    assert "n_echoes" not in out  # no input to count from
