"""Tests for ``src/mriforge/data/adapters/spatial.py``.

Covers the spatial-rank adapter trio:

* ``slice_3d_to_2d``        — pre_model, folds depth into batch (existing)
* ``flatten_spatial_to_1d`` — pre_model, flattens H×W into a sequence (F2 / 2026-05-20)
* ``gather_2d_to_3d``       — post_model, inverse of ``slice_3d_to_2d``
"""

from __future__ import annotations

import pytest
import torch

# Force adapter registration.
import mriforge.data.adapters  # noqa: F401
from mriforge.data.adapters.registry import ADAPTER_REGISTRY
from mriforge.data.adapters.spatial import (
    FlattenSpatialTo1D,
    Gather2Dto3D,
    Slice3Dto2D,
    Unflatten1DtoSpatial,
)


def test_flatten_spatial_to_1d_registered() -> None:
    """The adapter is registered with the audit capability dict."""
    entry = ADAPTER_REGISTRY.get("flatten_spatial_to_1d")
    assert entry is not None
    caps = entry["capabilities"]
    assert caps.bridges_from == {"spatial_dims": 2, "domain": "image"}
    assert caps.bridges_to == {"spatial_dims": 1, "domain": "image"}
    assert "pre_model" in caps.insertion_points


def test_flatten_spatial_to_1d_forward_2d_to_1d() -> None:
    """``[B, C, H, W]`` → ``[B, C, H*W]``."""
    x = torch.randn(2, 3, 4, 5)
    out = FlattenSpatialTo1D()(x)
    assert out.shape == (2, 3, 20)


def test_flatten_spatial_to_1d_forward_already_1d_identity() -> None:
    """3-D inputs already shaped ``[B, C, L]`` pass through unchanged."""
    x = torch.randn(2, 3, 7)
    out = FlattenSpatialTo1D()(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)


def test_flatten_spatial_to_1d_rejects_5d() -> None:
    """5-D tensors are rejected — caller must slice depth first."""
    x = torch.randn(2, 3, 4, 5, 6)
    with pytest.raises(ValueError, match="expects a 3-D or 4-D tensor"):
        FlattenSpatialTo1D()(x)


def test_slice_3d_to_2d_smoke() -> None:
    """Sanity check on the existing 3D→2D adapter.

    Default ``axis=2`` uses the depth-first ``[B, C, D, H, W]`` layout
    (MONAI convention) — D=4 here folds into batch, leaving ``[B*D, C, H, W]``.
    """
    x = torch.randn(2, 3, 4, 5, 6)  # [B, C, D, H, W] — depth-first
    out = Slice3Dto2D(axis=2)(x)
    assert out.shape == (2 * 4, 3, 5, 6)


def test_slice_3d_to_2d_depth_last_axis() -> None:
    """``axis=-1`` handles the depth-last ``[B, C, H, W, D]`` layout.

    ADAPT-003: the default ``axis=2`` assumes depth-first; depth-last
    callers must pass ``axis=-1`` (or ``axis=4``) to fold D into batch.
    """
    x = torch.randn(2, 3, 5, 6, 4)  # [B, C, H, W, D] — depth-last, D=4
    out = Slice3Dto2D(axis=-1)(x)
    assert out.shape == (2 * 4, 3, 5, 6)
    # axis=4 is equivalent to axis=-1 for a 5-D tensor.
    out4 = Slice3Dto2D(axis=4)(x)
    assert out4.shape == (2 * 4, 3, 5, 6)


def test_gather_2d_to_3d_smoke() -> None:
    """Sanity check on the existing 2D→3D companion adapter."""
    x = torch.randn(2 * 4, 3, 5, 6)
    out = Gather2Dto3D(axis=2)(x, depth=4)
    assert out.shape == (2, 3, 4, 5, 6)


def test_unflatten_1d_to_spatial_registered() -> None:
    """F-UNFLATTEN-1D / 2026-05-20 — companion to flatten_spatial_to_1d."""
    entry = ADAPTER_REGISTRY.get("unflatten_1d_to_spatial")
    assert entry is not None
    caps = entry["capabilities"]
    assert caps.bridges_from == {"spatial_dims": 1, "domain": "image"}
    assert caps.bridges_to == {"spatial_dims": 2, "domain": "image"}
    assert "post_model" in caps.insertion_points


def test_unflatten_1d_to_spatial_roundtrip() -> None:
    """Flatten then unflatten reproduces the original 4-D tensor."""
    x = torch.randn(2, 3, 4, 5)
    flat = FlattenSpatialTo1D()(x)
    assert flat.shape == (2, 3, 20)
    out = Unflatten1DtoSpatial()(flat, h=4, w=5)
    torch.testing.assert_close(out, x)


def test_unflatten_1d_to_spatial_rejects_bad_hw() -> None:
    """Mismatched H, W raises rather than silently reshaping."""
    x = torch.randn(2, 3, 20)
    with pytest.raises(ValueError, match=r"h\*w"):
        Unflatten1DtoSpatial()(x, h=3, w=5)  # 15 != 20


def test_unflatten_1d_to_spatial_rejects_non_3d() -> None:
    """4-D input is rejected (caller passed un-flattened tensor)."""
    x = torch.randn(2, 3, 4, 5)
    with pytest.raises(ValueError, match=r"3-D tensor"):
        Unflatten1DtoSpatial()(x, h=4, w=5)
