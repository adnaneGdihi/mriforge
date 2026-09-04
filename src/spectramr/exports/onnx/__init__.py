"""ONNX export helpers.

The single ONNX export SSOT for the framework. :class:`ONNXExporter` enforces
the no-dummy-input guard (CLAUDE.md #9), infers per-rank dynamic axes, rejects
complex inputs (ONNX has no complex dtype), and cleans up a partial file on
failure. The ``spectramr export`` CLI command dispatches through it.
"""

from spectramr.exports.onnx.onnx_exporter import ONNXExporter

__all__ = ["ONNXExporter"]
