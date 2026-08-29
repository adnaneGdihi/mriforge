"""Utility helpers for building ONNX dynamic axes mappings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch


def extract_tensor_shape(sample: Any) -> tuple[int, ...] | None:
    """Return the shape of the first tensor discovered in ``sample``."""

    if isinstance(sample, torch.Tensor):
        return tuple(sample.shape)

    if isinstance(sample, Mapping):
        values = cast(Iterable[Any], sample.values())
        for value in values:
            shape = extract_tensor_shape(value)
            if shape is not None:
                return shape
        return None

    if isinstance(sample, Iterable) and not isinstance(sample, (str, bytes)):
        sequence = cast(Iterable[Any], sample)
        for value in sequence:
            shape = extract_tensor_shape(value)
            if shape is not None:
                return shape
        return None

    return None


def _axes_for_shape(shape: tuple[int, ...]) -> dict[int, str]:
    """Build a dynamic-axes map for a tensor shape.

    Raises:
        ValueError: on a rank-0 (scalar) shape — there is no axis 0 to mark
            dynamic, so silently returning ``{0: "batch"}`` would emit an invalid
            dynamic-axes map (CLAUDE.md #9: no silent degradation).
    """

    rank = len(shape)
    if rank == 0:
        raise ValueError(
            "Cannot build ONNX dynamic axes for a rank-0 (scalar) tensor: "
            "there is no batch axis to make dynamic."
        )

    axes: dict[int, str] = {0: "batch"}

    if rank == 3:
        axes[2] = "length"
    elif rank == 4:
        axes[2] = "height"
        axes[3] = "width"
    elif rank >= 5:
        axes[2] = "depth"
        axes[3] = "height"
        axes[4] = "width"
        for idx in range(5, rank):
            axes[idx] = f"dim{idx}"

    return axes


def build_dynamic_axes(
    input_shape: tuple[int, ...] | None,
    output_shape: tuple[int, ...] | None,
) -> dict[str, dict[int, str]]:
    """Create a dynamic-axes dictionary for ONNX export."""

    dynamic_axes: dict[str, dict[int, str]] = {}

    if input_shape is not None:
        dynamic_axes["input"] = _axes_for_shape(input_shape)

    if output_shape is not None:
        dynamic_axes["output"] = _axes_for_shape(output_shape)

    if "input" not in dynamic_axes:
        dynamic_axes["input"] = {0: "batch"}

    if "output" not in dynamic_axes:
        dynamic_axes["output"] = {0: "batch"}

    return dynamic_axes


__all__ = ["build_dynamic_axes", "extract_tensor_shape"]
