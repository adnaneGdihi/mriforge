"""Tests for ``EdgeExporter`` and ``EdgeExportArtifact``.

Targets ``spectramr.infrastructure.inference.edge_export``. Multi-backend
exporter for edge deployment. We exercise:

- TorchScript export (always available)
- INT8 dynamic quantisation export (always available)
- Output-directory creation
- ``EdgeExportArtifact`` dataclass

ONNX export needs the full ONNX runtime + Mamba fallback path —
exercised in integration tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.inference.edge_export import EdgeExportArtifact, EdgeExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Trivial model: one Linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# EdgeExportArtifact
# ---------------------------------------------------------------------------


def test_artifact_dataclass_persists_fields(tmp_path: Path) -> None:
    """``EdgeExportArtifact`` stores backend, path, input shape, notes."""
    artifact = EdgeExportArtifact(
        backend="torchscript",
        path=tmp_path / "model.ts",
        input_shape=(1, 8),
        notes="test",
    )
    assert artifact.backend == "torchscript"
    assert artifact.path == tmp_path / "model.ts"
    assert artifact.input_shape == (1, 8)
    assert artifact.notes == "test"


def test_artifact_default_notes_empty() -> None:
    """Default ``notes`` is empty string."""
    artifact = EdgeExportArtifact(
        backend="onnx", path=Path("/tmp/x.onnx"), input_shape=(1,)
    )
    assert artifact.notes == ""


# ---------------------------------------------------------------------------
# EdgeExporter — construction
# ---------------------------------------------------------------------------


def test_constructor_creates_output_dir(tmp_path: Path) -> None:
    """``__init__`` creates the output directory if missing."""
    out_dir = tmp_path / "exports" / "nested"
    EdgeExporter(_SimpleModel(), torch.zeros(1, 8), out_dir)
    assert out_dir.exists()
    assert out_dir.is_dir()


def test_constructor_persists_attributes(tmp_path: Path) -> None:
    """Constructor stores model, sample_input, output_dir."""
    model = _SimpleModel()
    sample = torch.zeros(1, 8)
    exporter = EdgeExporter(model, sample, tmp_path)
    assert exporter.model is model
    assert torch.equal(exporter.sample_input, sample)
    assert exporter.output_dir == tmp_path


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------


def test_torchscript_export_writes_file(tmp_path: Path) -> None:
    """TorchScript export writes a file at the requested path."""
    model = _SimpleModel()
    sample = torch.randn(2, 8)
    exporter = EdgeExporter(model, sample, tmp_path)
    artifact = exporter.export_torchscript("model.ts")
    assert artifact.path.exists()
    assert artifact.backend == "torchscript"
    assert artifact.input_shape == (2, 8)


def test_torchscript_export_default_filename(tmp_path: Path) -> None:
    """Default filename is ``geomamba_ulf.ts``."""
    exporter = EdgeExporter(_SimpleModel(), torch.zeros(1, 8), tmp_path)
    artifact = exporter.export_torchscript()
    assert artifact.path.name == "geomamba_ulf.ts"


def test_torchscript_artifact_loadable(tmp_path: Path) -> None:
    """The exported TorchScript file can be loaded and run."""
    model = _SimpleModel()
    sample = torch.randn(2, 8)
    exporter = EdgeExporter(model, sample, tmp_path)
    artifact = exporter.export_torchscript()

    loaded = torch.jit.load(str(artifact.path))
    out = loaded(sample)
    assert out.shape == (2, 4)


def test_torchscript_export_puts_model_in_eval_mode(tmp_path: Path) -> None:
    """After export, the source model is in eval mode on CPU."""
    model = _SimpleModel().train()  # start in train mode
    exporter = EdgeExporter(model, torch.zeros(1, 8), tmp_path)
    exporter.export_torchscript()
    assert model.training is False
    for p in model.parameters():
        assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# INT8 dynamic quantisation
# ---------------------------------------------------------------------------


def test_int8_dynamic_export_writes_file(tmp_path: Path) -> None:
    """INT8 dynamic export writes a file at the requested path."""
    model = _SimpleModel()
    sample = torch.zeros(1, 8)
    exporter = EdgeExporter(model, sample, tmp_path)
    artifact = exporter.export_int8_dynamic("model_int8.pt")
    assert artifact.path.exists()
    assert artifact.backend == "int8_dynamic"


def test_int8_dynamic_default_filename(tmp_path: Path) -> None:
    """Default filename is ``geomamba_ulf_int8.pt``."""
    exporter = EdgeExporter(_SimpleModel(), torch.zeros(1, 8), tmp_path)
    artifact = exporter.export_int8_dynamic()
    assert artifact.path.name == "geomamba_ulf_int8.pt"


def test_int8_artifact_includes_notes(tmp_path: Path) -> None:
    """Int8 artifact ``notes`` mention quantize_dynamic."""
    exporter = EdgeExporter(_SimpleModel(), torch.zeros(1, 8), tmp_path)
    artifact = exporter.export_int8_dynamic()
    assert "quantize_dynamic" in artifact.notes
