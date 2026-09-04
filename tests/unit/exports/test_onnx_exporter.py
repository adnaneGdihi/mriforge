"""Unit tests for :class:`spectramr.exports.onnx.onnx_exporter.ONNXExporter`.

Regression coverage for the WS-8 fix: ``ONNXExporter.export`` used to fabricate a
``torch.zeros(1, 1)`` dummy when ``input_sample`` was ``None``, which traced the
model at the wrong shape and silently emitted a broken ONNX graph (CLAUDE.md #9 /
pitfall #9: no silent fallbacks). It must now raise ``ValueError`` instead, and a
real ``input_sample`` must get past that None-check.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from spectramr.exports.onnx.onnx_exporter import ONNXExporter

pytestmark = pytest.mark.unit


class _TrivialModule(torch.nn.Module):
    """Minimal identity-ish module so we never run a real ONNX export."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.linear(x)


def test_export_none_input_sample_raises_value_error(tmp_path) -> None:
    """``input_sample=None`` must raise ValueError, never fabricate a dummy."""

    exporter = ONNXExporter()
    model = _TrivialModule()
    out_path = tmp_path / "model.onnx"

    # Guard: even if the None-check regressed, a real torch.onnx.export must not
    # run here. Patch it so a regression surfaces as "did not raise" rather than
    # an unrelated tracing error.
    with patch("torch.onnx.export") as mock_export:
        with pytest.raises(ValueError, match="input_sample is required"):
            exporter.export(model, out_path, input_sample=None)

    mock_export.assert_not_called()
    # The fabricated-dummy path used to write/touch a file; nothing should land.
    assert not out_path.exists()


def test_export_with_real_input_passes_none_check(tmp_path) -> None:
    """A real input tensor gets past the None-check and reaches torch.onnx.export."""

    exporter = ONNXExporter()
    model = _TrivialModule()
    out_path = tmp_path / "model.onnx"
    sample = torch.zeros(2, 4)

    with patch("torch.onnx.export") as mock_export:
        returned = exporter.export(model, out_path, input_sample=sample)

    # Past the None-check: the real exporter was invoked exactly once.
    mock_export.assert_called_once()
    # The model is the first positional arg; the traced args tuple is the second.
    call_args = mock_export.call_args.args
    assert call_args[0] is model
    assert isinstance(call_args[1], tuple)
    assert call_args[1][0] is sample
    # export() returns the written path (touched after the patched export).
    assert returned == out_path
    assert out_path.exists()


def test_export_sequence_input_passes_none_check(tmp_path) -> None:
    """A tuple of tensors is also a valid (non-None) sample and traces through."""

    exporter = ONNXExporter()
    model = _TrivialModule()
    out_path = tmp_path / "model.onnx"
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)

    with patch("torch.onnx.export") as mock_export:
        exporter.export(model, out_path, input_sample=(a, b))

    mock_export.assert_called_once()
    traced_args = mock_export.call_args.args[1]
    assert isinstance(traced_args, tuple)
    assert traced_args == (a, b)


# ---------------------------------------------------------------------------
# WS-C C5: complex-dtype guard. ONNX has no complex type, so a complex input
# must fail loud (no silently-broken graph) rather than reaching torch.onnx.
# ---------------------------------------------------------------------------


def test_export_complex_tensor_raises(tmp_path) -> None:
    exporter = ONNXExporter()
    model = _TrivialModule()
    out_path = tmp_path / "model.onnx"
    complex_sample = torch.zeros(2, 4, dtype=torch.complex64)

    with patch("torch.onnx.export") as mock_export:
        with pytest.raises(ValueError, match="complex"):
            exporter.export(model, out_path, input_sample=complex_sample)

    mock_export.assert_not_called()
    assert not out_path.exists()


def test_export_complex_tensor_in_sequence_raises(tmp_path) -> None:
    exporter = ONNXExporter()
    model = _TrivialModule()
    out_path = tmp_path / "model.onnx"
    real = torch.zeros(2, 4)
    cplx = torch.zeros(2, 4, dtype=torch.complex64)

    with patch("torch.onnx.export") as mock_export:
        with pytest.raises(ValueError, match="complex"):
            exporter.export(model, out_path, input_sample=(real, cplx))

    mock_export.assert_not_called()
