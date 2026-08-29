"""Tests for ``SlabToChannel`` and ``SelectMiddleSlice``.

Targets ``mriforge.data.transforms.slab_transforms``. Two TorchIO
transforms used for 2.5D slicing of 3D volumes.

Categories:

- ``SlabToChannel`` flattens depth into channels: ``[C, W, H, D] →
  [C·D, W, H, 1]``
- ``SlabToChannel`` is a no-op when D = 1 (already-2D slices)
- ``SelectMiddleSlice`` reduces depth to 1, keeping the middle index
- Both transforms only touch the configured ``include`` keys
- Output volume preserves the W and H dimensions
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchio as tio

from mriforge.data.transforms.slab_transforms import (
    SelectMiddleSlice,
    SlabToChannel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subject(
    input_data: torch.Tensor,
    target_data: torch.Tensor | None = None,
) -> tio.Subject:
    """Build a TorchIO subject with input + target."""
    affine = np.eye(4)
    kwargs = {"input": tio.ScalarImage(tensor=input_data, affine=affine)}
    if target_data is not None:
        kwargs["target"] = tio.ScalarImage(tensor=target_data, affine=affine)
    return tio.Subject(**kwargs)


# ---------------------------------------------------------------------------
# SlabToChannel
# ---------------------------------------------------------------------------


def test_slab_to_channel_flattens_depth_into_channels() -> None:
    """``[C, W, H, D]`` collapses to ``[C·D, W, H, 1]``."""
    C, W, H, D = 2, 8, 8, 4
    data = torch.randn(C, W, H, D)
    subj = _make_subject(data)
    transform = SlabToChannel(include=("input",))
    out = transform.apply_transform(subj)["input"].data
    assert out.shape == (C * D, W, H, 1)


def test_slab_to_channel_noop_when_depth_is_one() -> None:
    """If D = 1 the data is already 2D; transform leaves it unchanged."""
    data = torch.randn(2, 8, 8, 1)
    subj = _make_subject(data)
    transform = SlabToChannel(include=("input",))
    out = transform.apply_transform(subj)["input"].data
    assert out.shape == (2, 8, 8, 1)


def test_slab_to_channel_only_touches_included_keys() -> None:
    """Keys not in ``include`` are passed through untouched."""
    data_in = torch.randn(2, 8, 8, 4)
    data_tgt = torch.randn(2, 8, 8, 4)
    subj = _make_subject(data_in, data_tgt)
    transform = SlabToChannel(include=("input",))
    out = transform.apply_transform(subj)
    # Input was reshaped, target was not.
    assert out["input"].data.shape == (8, 8, 8, 1)
    assert out["target"].data.shape == (2, 8, 8, 4)


# ---------------------------------------------------------------------------
# SelectMiddleSlice
# ---------------------------------------------------------------------------


def test_select_middle_slice_reduces_depth_to_one() -> None:
    """``D`` collapses to 1, picking the middle slice."""
    C, W, H, D = 2, 8, 8, 9
    # Tag each depth slice with its index so we can verify which was picked.
    data = torch.zeros(C, W, H, D)
    for d in range(D):
        data[..., d] = float(d)
    subj = _make_subject(data)
    transform = SelectMiddleSlice(include=("input",))
    out = transform.apply_transform(subj)["input"].data
    assert out.shape == (C, W, H, 1)
    # D=9 → mid_idx=4.
    assert torch.all(out[..., 0] == 4.0)


def test_select_middle_slice_only_touches_included_keys() -> None:
    """Keys outside ``include`` are passed through."""
    data_in = torch.randn(2, 8, 8, 4)
    data_tgt = torch.randn(2, 8, 8, 4)
    subj = _make_subject(data_in, data_tgt)
    # Default include is ("target",) — so input should pass through.
    transform = SelectMiddleSlice()
    out = transform.apply_transform(subj)
    assert out["target"].data.shape == (2, 8, 8, 1)
    assert out["input"].data.shape == (2, 8, 8, 4)


def test_select_middle_slice_preserves_spatial_dims() -> None:
    """W and H survive unchanged through the transform."""
    data = torch.randn(2, 16, 32, 10)
    subj = _make_subject(target_data=data, input_data=torch.zeros_like(data))
    transform = SelectMiddleSlice(include=("target",))
    out = transform.apply_transform(subj)["target"].data
    assert out.shape == (2, 16, 32, 1)
