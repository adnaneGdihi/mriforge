"""Device handling utilities for consistent tensor and model device management."""

import logging

import torch

logger = logging.getLogger(__name__)


def ensure_device_consistency(
    model: torch.nn.Module,
    device: str | torch.device,
) -> torch.nn.Module:
    """Ensure all model parameters and buffers are on the specified device.

    Args:
        model: PyTorch model to move
        device: Target device (str or torch.device)

    Returns:
        Model moved to the target device

    """
    if isinstance(device, str):
        device = torch.device(device)

    # Move the model to device
    model = model.to(device)

    # Ensure all parameters are on the target device
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)

    # Ensure all buffers are on the target device
    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)

    return model


def safe_to_device(
    tensor_or_model: torch.Tensor | torch.nn.Module,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | torch.nn.Module:
    """Safely move tensor or model to device with error handling.

    Args:
        tensor_or_model: Tensor or model to move
        device: Target device
        dtype: Optional target dtype

    Returns:
        Moved tensor or model

    """
    try:
        if tensor_or_model is None:
            return tensor_or_model

        if isinstance(device, str):
            device = torch.device(device)

        if hasattr(tensor_or_model, "to"):
            if hasattr(tensor_or_model, "parameters"):
                # This is a model
                return ensure_device_consistency(tensor_or_model, device)
            # This is a tensor
            if dtype is not None:
                return tensor_or_model.to(device=device, dtype=dtype)
            return tensor_or_model.to(device)

        return tensor_or_model

    except Exception as e:
        logger.warning(f"Could not move to device {device}: {e}")
        return tensor_or_model


def get_device_dtype_info(model: torch.nn.Module) -> dict:
    """Get information about device and dtype distribution in a model.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary with device and dtype statistics

    """
    devices: dict[str, int] = {}
    dtypes: dict[str, int] = {}

    for _name, param in model.named_parameters():
        device = str(param.device)
        dtype = str(param.dtype)

        devices[device] = devices.get(device, 0) + 1
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    return {
        "devices": devices,
        "dtypes": dtypes,
        "primary_device": (max(devices.keys(), key=devices.get) if devices else "unknown"),
        "primary_dtype": max(dtypes.keys(), key=dtypes.get) if dtypes else "unknown",
    }


def validate_device_consistency(
    model: torch.nn.Module,
    expected_device: str | torch.device,
) -> bool:
    """Validate that all model parameters are on the expected device.

    Args:
        model: Model to validate
        expected_device: Expected device

    Returns:
        True if all parameters are on expected device, False otherwise

    """
    if isinstance(expected_device, str):
        expected_device = torch.device(expected_device)

    for param in model.parameters():
        if param.device.type != expected_device.type:
            return False
        # If expected device has an index, check it matches
        if hasattr(expected_device, "index") and expected_device.index is not None:
            if param.device.index != expected_device.index:
                return False

    for buffer in model.buffers():
        if buffer.device.type != expected_device.type:
            return False
        # If expected device has an index, check it matches
        if hasattr(expected_device, "index") and expected_device.index is not None:
            if buffer.device.index != expected_device.index:
                return False

    return True


def create_device_aligned_tensor(
    shape: tuple,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
    fill_value: float | None = None,
) -> torch.Tensor:
    """Create a tensor with specific shape, device, and dtype.

    Args:
        shape: Tensor shape
        device: Target device
        dtype: Target dtype
        fill_value: Optional fill value (if None, uses zeros)

    Returns:
        Created tensor

    """
    if isinstance(device, str):
        device = torch.device(device)

    if fill_value is not None:
        tensor = torch.full(shape, fill_value, device=device, dtype=dtype)
    else:
        tensor = torch.zeros(shape, device=device, dtype=dtype)

    return tensor


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations.

    Returns:
        torch.device: CUDA device if available, otherwise CPU

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_memory_usage(device: torch.device, message: str = "") -> None:
    """Log current memory usage on the specified device.

    Args:
        device: Device to check memory usage for
        message: Optional message prefix for logging

    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        logger.debug(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
