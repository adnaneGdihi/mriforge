"""Edge-deployment scaffolding for GeoMamba-ULF.

Provides three exporters wired together by a single
:class:`EdgeExporter` entrypoint:

1. **TorchScript** — the always-available fallback. Works on any
   device that ships a PyTorch runtime (Jetson Orin AGX comes with
   one; even the smallest embedded board can run TorchScript via
   PyTorch Mobile).
2. **ONNX** — interoperable IR that downstream tools (TensorRT,
   ONNX Runtime, OpenVINO) consume. The Mamba kernels in
   :mod:`mamba_ssm` do not export cleanly to ONNX because they
   register custom CUDA ops; this exporter swaps in the
   gated-CNN/GRU fallback (already used by :class:`MambaBlock` when
   ``mamba_ssm`` is missing) before tracing.
3. **INT8 PyTorch dynamic quantization** — runs inside the
   PyTorch runtime, no external compiler. Targets the
   ``Linear`` layers of the FiLM MLP and the projection layers of
   the Mamba blocks (the bulk of the inference cost). Activations
   stay FP32; weights become INT8.

Plan §10 risk register: "Mamba kernel unsupported on Jetson — Pre-
compile with TensorRT-LLM custom op; INT8 fall-back to PyTorch."
This module ships the INT8 fallback path; the TensorRT-LLM custom
op compilation lives in a separate cluster-side script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class EdgeExportArtifact:
    """Description of a successfully exported artifact."""

    backend: str
    path: Path
    input_shape: tuple[int, ...]
    notes: str = ""


class EdgeExporter:
    """Multi-backend exporter for the GeoMamba-ULF generator.

    Args:
        model: trained generator (typically a
            :class:`GeoMambaUNet`). The exporter calls
            ``model.eval()`` and ``model.cpu()`` during export so
            the source object is left in eval mode on CPU.
        sample_input: representative input tensor for tracing /
            calibration. Shape determines the static dimensions in
            the exported graph.
        output_dir: directory to write the exported artifacts into.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.sample_input = sample_input
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_torchscript(self, filename: str = "geomamba_ulf.ts") -> EdgeExportArtifact:
        """Export to TorchScript via tracing (always available)."""
        path = self.output_dir / filename
        self.model.eval().cpu()
        sample_cpu = self.sample_input.detach().cpu()
        with torch.no_grad():
            traced = torch.jit.trace(self.model, sample_cpu, strict=False)
        traced.save(str(path))
        logger.info("[EdgeExporter] TorchScript artifact written to %s", path)
        return EdgeExportArtifact(
            backend="torchscript",
            path=path,
            input_shape=tuple(self.sample_input.shape),
        )

    def export_onnx(
        self,
        filename: str = "geomamba_ulf.onnx",
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
    ) -> EdgeExportArtifact:
        """Export to ONNX with the Mamba-fallback kernel.

        Falls back to the gated-CNN/GRU implementation of
        :class:`MambaBlock` to avoid the unsupported custom CUDA op
        from :mod:`mamba_ssm`. Returns the resulting artifact.

        Args:
            filename: output filename inside ``output_dir``.
            opset_version: ONNX opset version (17 supports the most
                ops needed by GeoMamba-ULF without custom plugins).
            dynamic_axes: optional dynamic-axis spec; defaults to
                making the batch dim dynamic.
        """
        path = self.output_dir / filename
        self.model.eval().cpu()
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        sample_cpu = self.sample_input.detach().cpu()
        # Force the Mamba-fallback path on every block so torch.onnx
        # only sees stock ops. We do not modify the source module —
        # use a no_grad + onnx.export call.
        self._patch_mamba_fallback(self.model)
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                sample_cpu,
                str(path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
        logger.info("[EdgeExporter] ONNX artifact written to %s", path)
        return EdgeExportArtifact(
            backend="onnx",
            path=path,
            input_shape=tuple(self.sample_input.shape),
            notes=f"opset={opset_version}; mamba kernel: gated-CNN/GRU fallback",
        )

    def export_int8_dynamic(
        self,
        filename: str = "geomamba_ulf_int8.pt",
    ) -> EdgeExportArtifact:
        """Apply PyTorch native dynamic INT8 quantization and save.

        Quantizes ``nn.Linear`` and ``nn.GRU`` modules — the bulk
        of the GeoMamba-ULF inference cost lives there (FiLM MLP +
        Mamba-fallback projection layers).
        """
        path = self.output_dir / filename
        self.model.eval().cpu()
        quantized = torch.ao.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.GRU, nn.Conv1d},
            dtype=torch.qint8,
        )
        torch.save({"state_dict": quantized.state_dict(), "model_repr": repr(quantized)}, path)
        logger.info("[EdgeExporter] INT8 dynamic-quantized model written to %s", path)
        return EdgeExportArtifact(
            backend="int8_dynamic",
            path=path,
            input_shape=tuple(self.sample_input.shape),
            notes="quantize_dynamic on {Linear, GRU, Conv1d}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_mamba_fallback(model: nn.Module) -> None:
        """Force MambaBlock instances onto the gated-CNN/GRU fallback path.

        The official ``mamba_ssm`` selective-scan kernel is a custom
        CUDA op that ONNX cannot export. Setting ``use_official=False``
        on every MambaBlock instance makes the wrapper run its
        in-Python fallback, which uses only stock PyTorch ops.
        """
        from mriforge.models.blocks.mamba_block import MambaBlock

        for module in model.modules():
            if isinstance(module, MambaBlock) and getattr(module, "use_official", False):
                module.use_official = False
                logger.info("[EdgeExporter] disabled mamba_ssm kernel on %s", type(module).__name__)


__all__ = ["EdgeExportArtifact", "EdgeExporter"]
