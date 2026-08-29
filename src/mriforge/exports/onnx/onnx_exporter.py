"""Utility for exporting PyTorch models to ONNX."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from mriforge.shared.utils.onnx_dynamic_axes import build_dynamic_axes, extract_tensor_shape


def _reject_complex_inputs(sample: Any) -> None:
    """Raise if any tensor reachable in ``sample`` has a complex dtype.

    ONNX cannot represent complex tensors; exporting one silently produces a
    broken graph. MRI k-space must be converted to a real/interleaved-channel
    layout before export.
    """
    if isinstance(sample, torch.Tensor):
        if sample.is_complex():
            raise ValueError(
                "ONNX export received a complex-dtype input tensor "
                f"({sample.dtype}); ONNX has no complex type. Convert complex "
                "k-space to a 2-channel real layout before export."
            )
        return
    if isinstance(sample, Mapping):
        for value in sample.values():
            _reject_complex_inputs(value)
        return
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        for value in sample:
            _reject_complex_inputs(value)


class ONNXExporter:
    """Small convenience wrapper for :func:`torch.onnx.export`."""

    def __init__(
        self,
        opset_version: int = 17,
        dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    ) -> None:
        """__init__.

        Args:
            opset_version (int): Description.
            dynamic_axes (Mapping[str, Mapping[int, str]] | None): Description.
        """
        self.opset_version = opset_version
        if dynamic_axes is None:
            self.dynamic_axes: dict[str, dict[int, str]] | None = None
        else:
            self.dynamic_axes = {name: dict(axes) for name, axes in dynamic_axes.items()}

    def export(
        self,
        model: torch.nn.Module,
        output_path: str | Path,
        input_sample: torch.Tensor | Sequence[Any] | None = None,
        output_sample: torch.Tensor | Sequence[Any] | None = None,
        **export_kwargs: Any,
    ) -> Path:
        """Export ``model`` to ONNX and return the written path."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(model, torch.nn.Module):
            msg = "model must be an instance of torch.nn.Module"
            raise TypeError(msg)

        model.eval()

        # No-silent-fallback (CLAUDE.md #9): a fabricated (1, 1) dummy would
        # trace the model at the wrong input shape and silently emit a broken
        # ONNX graph. Require a real sample to trace against.
        if input_sample is None:
            raise ValueError(
                "input_sample is required to trace the model for ONNX export; "
                "pass a representative input tensor (or tuple of tensors)."
            )
        sample_for_export: torch.Tensor | Sequence[Any] = input_sample

        if isinstance(sample_for_export, torch.Tensor):
            export_args: tuple[Any, ...] = (sample_for_export,)
        else:
            export_args = tuple(sample_for_export)

        # ONNX has no complex dtype: a complex input traces to a broken graph
        # (or a cryptic torch.onnx error). Fail loud with an actionable message
        # (CLAUDE.md #9). MRI models here operate on real/interleaved-channel
        # tensors, so convert complex k-space to a 2-channel real layout upstream.
        _reject_complex_inputs(export_args)

        input_shape = extract_tensor_shape(export_args)
        output_shape = extract_tensor_shape(output_sample)
        dynamic_axes = self.dynamic_axes or build_dynamic_axes(
            input_shape,
            output_shape,
        )

        input_names = list(export_kwargs.setdefault("input_names", ["input"]))
        output_names = list(export_kwargs.setdefault("output_names", ["output"]))

        for name in input_names:
            if name not in dynamic_axes:
                dynamic_axes[name] = dict(dynamic_axes.get("input", {0: "batch"}))

        for name in output_names:
            if name not in dynamic_axes:
                dynamic_axes[name] = dict(dynamic_axes.get("output", {0: "batch"}))

        try:
            torch.onnx.export(
                model,
                export_args,
                output_path.as_posix(),
                opset_version=self.opset_version,
                dynamic_axes=dynamic_axes,
                **export_kwargs,
            )  # type: ignore[misc]
        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise

        output_path.touch(exist_ok=True)
        return output_path
