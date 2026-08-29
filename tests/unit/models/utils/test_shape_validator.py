"""Unit tests for models/utils/shape_validator.py.

Covers: validate_tensor_shape, validate_5d_or_4d_tensor,
        validate_channel_count, validate_spatial_dims,
        validate_batch_size, assert_shape_equals.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.utils.shape_validator import (
    assert_shape_equals,
    validate_5d_or_4d_tensor,
    validate_batch_size,
    validate_channel_count,
    validate_spatial_dims,
    validate_tensor_shape,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_tensor_shape_canary():
    """4D tensor passes expected_dims=4 without error."""
    x = torch.zeros(2, 3, 64, 64)
    validate_tensor_shape(x, expected_dims=4)  # should not raise


# ---------------------------------------------------------------------------
# Parametrized: validate_tensor_shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(2, 1, 32, 32), (4, 3, 64, 64), (1, 2, 128, 128)])
def test_validate_tensor_shape_4d_ok(shape):
    x = torch.zeros(*shape)
    validate_tensor_shape(x, expected_dims=4)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 3, 4)])
def test_validate_tensor_shape_wrong_ndim_raises(shape):
    x = torch.zeros(*shape)
    with pytest.raises(ValueError, match="Expected 4D"):
        validate_tensor_shape(x, expected_dims=4)


# ---------------------------------------------------------------------------
# Parametrized: validate_5d_or_4d_tensor
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "shape,expect_5d",
    [
        ((2, 1, 32, 32), False),
        ((2, 1, 16, 16, 16), True),
    ],
)
def test_validate_5d_or_4d(shape, expect_5d):
    x = torch.zeros(*shape)
    is_5d, orig_shape = validate_5d_or_4d_tensor(x)
    assert is_5d is expect_5d
    assert orig_shape == x.shape


@pytest.mark.unit
def test_validate_5d_or_4d_rejects_3d():
    with pytest.raises(ValueError):
        validate_5d_or_4d_tensor(torch.zeros(2, 3, 4))


# ---------------------------------------------------------------------------
# Parametrized: validate_channel_count
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("expected", [1, 2, (1, 2)])
def test_validate_channel_count_ok(expected):
    x = torch.zeros(2, 1, 32, 32)  # C=1
    if expected == 2:
        x = torch.zeros(2, 2, 32, 32)
    elif expected == (1, 2):
        x = torch.zeros(2, 1, 32, 32)
    validate_channel_count(x, expected_channels=expected)


@pytest.mark.unit
def test_validate_channel_count_wrong_raises():
    x = torch.zeros(2, 3, 32, 32)
    with pytest.raises(ValueError, match="channel"):
        validate_channel_count(x, expected_channels=1)


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_tensor_shape_with_none_dims():
    """None entries in expected_shape are wildcard (any size)."""
    x = torch.zeros(2, 7, 32, 32)
    validate_tensor_shape(x, expected_shape=(2, None, 32, 32))


@pytest.mark.unit
def test_validate_batch_size_min():
    """Batch size of 1 passes min_batch_size=1."""
    x = torch.zeros(1, 2, 32, 32)
    validate_batch_size(x, min_batch_size=1)


@pytest.mark.unit
def test_assert_shape_equals_mismatch_raises_assertion_error():
    x = torch.zeros(2, 3, 32, 32)
    with pytest.raises(AssertionError):
        assert_shape_equals(x, (2, 3, 64, 64))


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_channel_count_1d_raises():
    """1D tensor (no channel dim) must raise ValueError."""
    x = torch.zeros(5)
    with pytest.raises(ValueError, match="at least 2D"):
        validate_channel_count(x, expected_channels=1)
