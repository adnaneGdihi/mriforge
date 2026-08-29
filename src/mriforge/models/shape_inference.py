"""Shape Inference Utilities for Convolutional Neural Networks

This module provides analytical shape inference utilities for convolutional
stacks, eliminating the need for runtime dummy tensor forward passes.

Features:
- Analytical computation of output shapes for conv2d and maxpool2d layers
- Support for common layer configurations
- Pre-defined layer stacks for existing models
- Efficient computation without requiring actual tensor operations

Usage:
    from core.models.shape_inference import infer_conv_stack_output

    # Infer output shape for a stack of layers
    h_out, w_out, c_out = infer_conv_stack_output(h=64, w=64, layers=layers)
"""

from typing import Any


def infer_conv2d_output_size(
    h_in: int,
    w_in: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> tuple[int, int]:
    """Analytically compute output size of a 2D convolution.

    Args:
        h_in: Input height
        w_in: Input width
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding applied to input
        dilation: Dilation factor

    Returns:
        Tuple of (output_height, output_width)

    """
    h_out = ((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    w_out = ((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    return h_out, w_out


def infer_maxpool2d_output_size(
    h_in: int,
    w_in: int,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
) -> tuple[int, int]:
    """Analytically compute output size of a 2D max pooling.

    Args:
        h_in: Input height
        w_in: Input width
        kernel_size: Size of the pooling kernel
        stride: Stride of the pooling (defaults to kernel_size)
        padding: Padding applied to input

    Returns:
        Tuple of (output_height, output_width)

    """
    if stride is None:
        stride = kernel_size

    h_out = ((h_in + 2 * padding - kernel_size) // stride) + 1
    w_out = ((w_in + 2 * padding - kernel_size) // stride) + 1
    return h_out, w_out


def infer_conv_stack_output(
    h: int,
    w: int,
    layers: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Infer output shape for a stack of convolutional and pooling layers.

    Args:
        h: Input height
        w: Input width
        layers: list of layer configurations

    Returns:
        Tuple of (output_height, output_width, output_channels)

    """
    h_current, w_current = h, w
    if layers:
        first = layers[0]
        # Only conv2d layers carry in_channels; for non-conv first layers we
        # default to 1 channel (caller is expected to track channels explicitly).
        if first.get("type") == "conv2d":
            if "in_channels" not in first:
                raise ValueError("First conv2d layer in the stack is missing 'in_channels'.")
            c_current = first["in_channels"]
        else:
            c_current = first.get("in_channels", 1)
    else:
        c_current = 1

    for layer in layers:
        if "type" not in layer:
            raise ValueError(f"Layer config missing 'type' key: {layer}")
        layer_type = layer["type"]

        if layer_type == "conv2d":
            h_current, w_current = infer_conv2d_output_size(
                h_current,
                w_current,
                kernel_size=layer["kernel_size"],
                stride=layer.get("stride", 1),
                padding=layer.get("padding", 0),
                dilation=layer.get("dilation", 1),
            )
            c_current = layer["out_channels"]

        elif layer_type == "maxpool2d":
            h_current, w_current = infer_maxpool2d_output_size(
                h_current,
                w_current,
                kernel_size=layer["kernel_size"],
                stride=layer.get("stride"),
                padding=layer.get("padding", 0),
            )
            # MaxPool doesn't change channel count

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    return h_current, w_current, c_current


def compute_flattened_size(h: int, w: int, c: int) -> int:
    """Compute the flattened size for a 3D tensor (C, H, W).

    Args:
        h: Height
        w: Width
        c: Channels

    Returns:
        Total number of elements when flattened

    """
    return h * w * c


def validate_conv2d_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> None:
    """Validate parameters for a 2D convolution layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding applied to input

    Raises:
        ValueError: If parameters are invalid

    """
    if in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if out_channels <= 0:
        raise ValueError(f"out_channels must be positive, got {out_channels}")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")


def validate_batch_size(batch_size: int, max_batch_size: int = 1024) -> None:
    """Validate batch size parameters.

    Args:
        batch_size: Batch size to validate
        max_batch_size: Maximum allowed batch size

    Raises:
        ValueError: If batch size is invalid

    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size > max_batch_size:
        raise ValueError(f"batch_size {batch_size} exceeds maximum {max_batch_size}")


def validate_image_dimensions(
    height: int,
    width: int,
    min_dim: int = 1,
    max_dim: int = 4096,
) -> None:
    """Validate image dimensions.

    Args:
        height: Image height
        width: Image width
        min_dim: Minimum allowed dimension
        max_dim: Maximum allowed dimension

    Raises:
        ValueError: If dimensions are invalid

    """
    if height < min_dim or height > max_dim:
        raise ValueError(f"height {height} must be between {min_dim} and {max_dim}")
    if width < min_dim or width > max_dim:
        raise ValueError(f"width {width} must be between {min_dim} and {max_dim}")


# Pre-defined layer configurations for existing models

WAVELET_KAN_ENCODER_LAYERS = [
    {
        "type": "conv2d",
        "in_channels": 1,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 128,
        "out_channels": 256,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 256,
        "out_channels": 512,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
]

FOURIER_KAN_ENCODER_LAYERS = [
    {
        "type": "conv2d",
        "in_channels": 1,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 64,
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 128,
        "out_channels": 256,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "type": "maxpool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    {
        "type": "conv2d",
        "in_channels": 256,
        "out_channels": 512,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
]
