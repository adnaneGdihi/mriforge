"""Shape validation utilities for tensor operations.

This module provides helpers to validate tensor shapes and catch dimension
mismatches early before they cause cryptic downstream errors.
"""

import torch


def validate_tensor_shape(
    x: torch.Tensor,
    expected_dims: int | None = None,
    expected_shape: tuple[int | None, ...] | None = None,
    name: str = "tensor",
) -> None:
    """Validate tensor shape and dimensions.

    Args:
        x: Tensor to validate
        expected_dims: Expected number of dimensions (e.g., 4 for 4D)
        expected_shape: Expected shape tuple. Use None for variable dimensions.
        name: Name of tensor for error messages

    Raises:
        ValueError: If shape doesn't match expectations

    Examples:
        >>> x = torch.randn(2, 3, 64, 64)
        >>> validate_tensor_shape(x, expected_dims=4, name="image")
        >>> validate_tensor_shape(x, expected_shape=(2, 3, 64, 64))
        >>> validate_tensor_shape(x, expected_shape=(2, None, 64, 64))  # Any C value
    """
    if expected_dims is not None:
        if x.dim() != expected_dims:
            raise ValueError(
                f"{name}: Expected {expected_dims}D tensor, got {x.dim()}D with shape {x.shape}"
            )

    if expected_shape is not None:
        if len(expected_shape) != x.dim():
            raise ValueError(
                f"{name}: Expected {len(expected_shape)}D shape, got {x.dim()}D with shape {x.shape}"
            )

        for i, (expected, actual) in enumerate(zip(expected_shape, x.shape, strict=False)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name}: Dimension {i} mismatch. Expected {expected}, got {actual}. "
                    f"Full shape: {x.shape}"
                )


def validate_5d_or_4d_tensor(x: torch.Tensor, name: str = "tensor") -> tuple[bool, tuple[int, ...]]:
    """Validate and handle both 4D and 5D tensors.

    Args:
        x: Input tensor
        name: Name for error messages

    Returns:
        Tuple of (is_5d, original_shape)

    Raises:
        ValueError: If tensor is not 4D or 5D
    """
    if x.dim() == 4:
        # (B, C, H, W)
        validate_tensor_shape(x, expected_dims=4, name=name)
        return False, x.shape
    elif x.dim() == 5:
        # (B, C, H, W, D) or (B, C, D, H, W)
        validate_tensor_shape(x, expected_dims=5, name=name)
        return True, x.shape
    else:
        raise ValueError(f"{name}: Expected 4D or 5D tensor, got {x.dim()}D with shape {x.shape}")


def validate_channel_count(
    x: torch.Tensor,
    expected_channels: int | tuple[int, ...],
    name: str = "tensor",
) -> None:
    """Validate that tensor has expected number of channels.

    Args:
        x: Tensor with shape (B, C, ...) or (..., C, H, W)
        expected_channels: Expected channel count(s). Can be int or tuple of ints.
        name: Name for error messages

    Raises:
        ValueError: If channel count doesn't match
    """
    if x.dim() < 2:
        raise ValueError(
            f"{name}: Tensor must be at least 2D (has batch and channel dims), "
            f"got {x.dim()}D with shape {x.shape}"
        )

    actual_channels = x.shape[1]

    if isinstance(expected_channels, int):
        expected = [expected_channels]
    else:
        expected = list(expected_channels)

    if actual_channels not in expected:
        raise ValueError(
            f"{name}: Expected {expected_channels} channel(s), got {actual_channels}. "
            f"Shape: {x.shape}"
        )


def validate_spatial_dims(
    x: torch.Tensor,
    min_spatial_dims: int | None = None,
    expected_spatial_dims: tuple[int | None, ...] | None = None,
    name: str = "tensor",
) -> None:
    """Validate spatial dimensions (H, W for 4D, H, W, D for 5D).

    Args:
        x: 4D or 5D tensor
        min_spatial_dims: Minimum size for each spatial dimension (e.g., (16, 16, 1))
        expected_spatial_dims: Expected exact spatial dimensions
        name: Name for error messages

    Raises:
        ValueError: If spatial dims don't match
    """
    if x.dim() == 4 or x.dim() == 5:
        spatial_shape = x.shape[2:]
    else:
        raise ValueError(f"{name}: Expected 4D or 5D tensor, got {x.dim()}D with shape {x.shape}")

    if expected_spatial_dims is not None:
        for i, (expected, actual) in enumerate(
            zip(expected_spatial_dims, spatial_shape, strict=False)
        ):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name}: Spatial dimension {i} mismatch. "
                    f"Expected {expected}, got {actual}. Full shape: {x.shape}"
                )

    if min_spatial_dims is not None:
        for i, (min_size, actual) in enumerate(zip(min_spatial_dims, spatial_shape, strict=False)):
            if actual < min_size:
                raise ValueError(
                    f"{name}: Spatial dimension {i} too small. "
                    f"Expected at least {min_size}, got {actual}. Full shape: {x.shape}"
                )


def validate_batch_size(
    x: torch.Tensor,
    expected_batch_size: int | None = None,
    min_batch_size: int = 1,
    name: str = "tensor",
) -> None:
    """Validate batch size.

    Args:
        x: Tensor with batch as first dimension
        expected_batch_size: Exact expected batch size
        min_batch_size: Minimum batch size
        name: Name for error messages

    Raises:
        ValueError: If batch size doesn't match
    """
    if x.dim() < 1:
        raise ValueError(f"{name}: Tensor must be at least 1D, got {x.dim()}D")

    actual_batch = x.shape[0]

    if expected_batch_size is not None:
        if actual_batch != expected_batch_size:
            raise ValueError(
                f"{name}: Expected batch size {expected_batch_size}, got {actual_batch}. "
                f"Shape: {x.shape}"
            )

    if actual_batch < min_batch_size:
        raise ValueError(
            f"{name}: Batch size {actual_batch} is less than minimum {min_batch_size}. "
            f"Shape: {x.shape}"
        )


def assert_shape_equals(
    x: torch.Tensor,
    expected_shape: tuple[int, ...],
    name: str = "tensor",
) -> None:
    """Assert that tensor has exact shape match.

    Args:
        x: Tensor to check
        expected_shape: Expected shape tuple
        name: Name for error messages

    Raises:
        AssertionError: If shapes don't match
    """
    if x.shape != expected_shape:
        raise AssertionError(f"{name}: Shape mismatch. Expected {expected_shape}, got {x.shape}")


__all__ = [
    "assert_shape_equals",
    "validate_5d_or_4d_tensor",
    "validate_batch_size",
    "validate_channel_count",
    "validate_spatial_dims",
    "validate_tensor_shape",
]
