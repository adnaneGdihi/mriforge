"""Tests for the reporting pipeline end-to-end behaviour.

The task description references ``src/infrastructure/reporting/pipeline.py``,
which does not currently exist in the repo. The reporting subsystem that is
actually present today is :mod:`spectramr.infrastructure.reporting.advanced_reporting`,
whose :class:`TrainingReport.generate_complete_report` is the pipeline that
chains text-report + plots + CSV + JSON serialisation. These tests cover that
real pipeline so the coverage gain is real, and they will also opportunistically
test a future ``pipeline`` module if/when it lands by importing it through
``pytest.importorskip``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from spectramr.infrastructure.reporting.advanced_reporting import (
    TrainingReport,
    create_training_report,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_config():
    """A duck-typed stand-in for TrainingConfig that the report can introspect."""

    return SimpleNamespace(
        model_type="pix2pix",
        epochs=3,
        batch_size=4,
        learning_rate=2e-4,
    )


@pytest.fixture
def populated_report(tmp_path, fake_config):
    report = TrainingReport(experiment_name="unit_smoke", output_dir=tmp_path)
    report.start_training(fake_config)
    for epoch in range(3):
        report.log_epoch_metrics(
            epoch=epoch,
            g_loss=1.0 - 0.1 * epoch,
            d_loss=0.5 + 0.05 * epoch,
            psnr=24.0 + epoch,
            ssim=0.80 + 0.01 * epoch,
            lr=2e-4,
            grad_norm_g=1.1,
            grad_norm_d=0.9,
        )
    report.end_training()
    return report


# --------------------------------------------------------------------------- #
# Smoke: pipeline runs end-to-end and writes the documented artefacts
# --------------------------------------------------------------------------- #


def test_pipeline_generates_complete_report_artifacts(populated_report, tmp_path):
    """generate_complete_report() should write the four documented artefacts."""

    out = populated_report.generate_complete_report()

    assert isinstance(out, Path)
    assert out.exists()
    # Documented files produced by the pipeline (text + CSV + JSON).
    # The plot is written in a background subprocess so we don't assert on it
    # to keep the test deterministic and CPU-only.
    assert (out / "training_report.txt").exists()
    assert (out / "metrics_history.csv").exists()
    assert (out / "training_report.json").exists()


def test_pipeline_json_payload_round_trips(populated_report):
    """The JSON artefact must be valid JSON and contain the metric series."""

    out = populated_report.generate_complete_report()
    payload = json.loads((out / "training_report.json").read_text())

    assert payload["experiment_name"] == "unit_smoke"
    assert "metrics_history" in payload
    assert payload["metrics_history"]["epoch"] == [0, 1, 2]
    assert payload["final_metrics"]["psnr"] == pytest.approx(26.0)


# --------------------------------------------------------------------------- #
# Empty input handling
# --------------------------------------------------------------------------- #


def test_pipeline_handles_empty_metrics(tmp_path, fake_config):
    """A pipeline run with no epoch metrics must not crash."""

    report = TrainingReport(experiment_name="empty", output_dir=tmp_path)
    report.start_training(fake_config)
    report.end_training()

    out = report.generate_complete_report()

    assert out.exists()
    assert (out / "training_report.txt").exists()
    # Text report has the documented sentinel branch when metrics_history is empty.
    text = (out / "training_report.txt").read_text()
    assert "TRAINING REPORT" in text


def test_pipeline_text_report_returns_sentinel_when_not_started(tmp_path):
    """Calling the text report generator before start_training returns the
    documented sentinel string, not an exception."""

    report = TrainingReport(experiment_name="never_started", output_dir=tmp_path)
    assert report.generate_training_progress_report() == "Training not started"


# --------------------------------------------------------------------------- #
# Missing-dependency handling: plotting is skipped when no metrics exist
# --------------------------------------------------------------------------- #


def test_pipeline_skips_plots_when_no_metrics(tmp_path, fake_config):
    """generate_plots() must early-return rather than spawning a worker when
    metrics_history is empty."""

    report = TrainingReport(experiment_name="no_plots", output_dir=tmp_path)
    report.start_training(fake_config)
    # Don't log any epochs.

    # generate_plots() early-returns when metrics_history["epoch"] is empty,
    # so no subprocess is spawned and no PNG is written.
    report.generate_plots()
    assert not (report.output_dir / "training_progress.png").exists()


def test_pipeline_factory_returns_report_with_started_state(tmp_path, fake_config):
    """create_training_report() must return a started TrainingReport."""

    report = create_training_report(
        experiment_name="from_factory",
        config=fake_config,
        output_dir=tmp_path,
    )
    assert isinstance(report, TrainingReport)
    assert report.start_time is not None
    assert "epoch" in report.metrics_history


# --------------------------------------------------------------------------- #
# Forward-compatible: if a future pipeline module is added, basic smoke
# --------------------------------------------------------------------------- #


def test_optional_future_pipeline_module_smoke(tmp_path):
    """If ``spectramr.infrastructure.reporting.pipeline`` is ever introduced, run a
    minimal import check so the test stays in lockstep with the planned API."""

    pipeline = pytest.importorskip("spectramr.infrastructure.reporting.pipeline")
    # We don't pin a specific API; we only assert that the module exposes some
    # callable entry-point (function or class) so it isn't an empty stub.
    public_attrs = [a for a in dir(pipeline) if not a.startswith("_")]
    assert public_attrs, "pipeline module should expose at least one public attr"
