"""Tests for the per-experiment metrics report generator.

The task targets ``metrics_report_generator.py``, which is not present in this
repo revision. The closest extant generator is
:meth:`TrainingReport.generate_training_progress_report`, which renders a
plain-text report (section headers, ruler lines and inline metric values) plus
:meth:`TrainingReport.save_report_json` which serialises the same data to JSON.
We exercise both, including determinism of the JSON serialiser.

A ``pytest.importorskip`` guard keeps the suite forward-compatible with a
future ``metrics_report_generator`` module.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest

from spectramr.infrastructure.reporting.advanced_reporting import TrainingReport

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def config_ns():
    return SimpleNamespace(
        model_type="cyclegan",
        epochs=2,
        batch_size=2,
        learning_rate=1e-4,
    )


@pytest.fixture
def report_with_metrics(tmp_path, config_ns):
    report = TrainingReport(experiment_name="gen_check", output_dir=tmp_path)
    report.start_training(config_ns)
    report.log_epoch_metrics(0, g_loss=1.0, d_loss=0.5, psnr=23.5, ssim=0.81, lr=1e-4)
    report.log_epoch_metrics(1, g_loss=0.8, d_loss=0.6, psnr=24.7, ssim=0.83, lr=1e-4)
    report.end_training()
    return report


# --------------------------------------------------------------------------- #
# Smoke: report renders on synthetic per-arm metrics dict
# --------------------------------------------------------------------------- #


def test_progress_report_renders_full_text(report_with_metrics):
    text = report_with_metrics.generate_training_progress_report()
    assert isinstance(text, str)
    assert text.strip(), "rendered report must not be empty"
    # Expected section headers documented in the implementation:
    expected_headers = (
        "TRAINING REPORT",
        "TRAINING SUMMARY",
        "FINAL METRICS",
        "BEST METRICS",
    )
    for header in expected_headers:
        assert header in text, f"missing section header {header!r}"


def test_progress_report_includes_config_introspection(report_with_metrics):
    text = report_with_metrics.generate_training_progress_report()
    # Values from the SimpleNamespace config must reach the report via getattr.
    assert "cyclegan" in text
    assert "Epochs: 2" in text
    assert "Batch Size: 2" in text


def test_progress_report_includes_final_and_best_metrics(report_with_metrics):
    text = report_with_metrics.generate_training_progress_report()
    # Final PSNR is the last logged value (24.7), best PSNR is also 24.7 here.
    assert "24.70" in text
    # Best metric line is annotated with the epoch index.
    assert re.search(r"Best PSNR:\s+24\.70 dB \(Epoch 1\)", text)


# --------------------------------------------------------------------------- #
# JSON output is parseable (round-trip via json module)
# --------------------------------------------------------------------------- #


def test_json_artifact_round_trips(report_with_metrics):
    report_with_metrics.save_report_json()
    payload = json.loads(
        (report_with_metrics.output_dir / "training_report.json").read_text(),
    )

    assert payload["experiment_name"] == "gen_check"
    assert payload["metrics_history"]["epoch"] == [0, 1]
    assert payload["final_metrics"]["psnr"] == pytest.approx(24.7)
    assert payload["final_metrics"]["ssim"] == pytest.approx(0.83)


# --------------------------------------------------------------------------- #
# Determinism: same input -> byte-identical JSON output
# --------------------------------------------------------------------------- #


def test_json_serialisation_is_deterministic(tmp_path, config_ns):
    """Byte-identical metrics_history must serialise to byte-identical JSON
    (sans the ``timestamp`` field, which is intentionally wall-clock based)."""

    def build_payload():
        report = TrainingReport(experiment_name="det", output_dir=tmp_path)
        report.start_training(config_ns)
        report.log_epoch_metrics(0, g_loss=1.0, d_loss=0.5, psnr=23.5, ssim=0.81)
        report.log_epoch_metrics(1, g_loss=0.8, d_loss=0.6, psnr=24.7, ssim=0.83)
        report.end_training()
        report.save_report_json()
        data = json.loads((report.output_dir / "training_report.json").read_text())
        # Strip wall-clock-dependent fields.
        data.pop("timestamp", None)
        data.pop("training_duration", None)
        return json.dumps(data, sort_keys=True)

    first = build_payload()
    second = build_payload()
    assert first == second


# --------------------------------------------------------------------------- #
# Missing metric must not raise KeyError
# --------------------------------------------------------------------------- #


def test_missing_optional_metric_does_not_raise(tmp_path, config_ns):
    """If PSNR/SSIM/LR aren't logged, the generator must default them rather
    than raising a KeyError. None values are coerced to 0.0 by design."""

    report = TrainingReport(experiment_name="partial", output_dir=tmp_path)
    report.start_training(config_ns)
    # Log only the two mandatory losses; everything else defaults.
    report.log_epoch_metrics(0, g_loss=1.0, d_loss=0.5)
    report.end_training()

    # Must not raise.
    text = report.generate_training_progress_report()
    report.save_report_json()
    payload = json.loads((report.output_dir / "training_report.json").read_text())

    # Documented placeholder: 0.0, never KeyError. The text report only shows
    # PSNR/SSIM when they are >0 — so the absence of those lines is the
    # documented placeholder behaviour.
    assert "PSNR" not in text.split("FINAL METRICS")[1].split("BEST METRICS")[0]
    assert payload["final_metrics"]["psnr"] == 0.0


def test_progress_report_with_no_epochs_yields_minimal_text(tmp_path, config_ns):
    """Zero logged epochs ⇒ the FINAL/BEST sections drop out cleanly."""

    report = TrainingReport(experiment_name="empty", output_dir=tmp_path)
    report.start_training(config_ns)
    report.end_training()
    text = report.generate_training_progress_report()

    assert "TRAINING REPORT" in text
    assert "FINAL METRICS" not in text
    assert "BEST METRICS" not in text


# --------------------------------------------------------------------------- #
# Forward compatibility
# --------------------------------------------------------------------------- #


def test_optional_future_metrics_report_generator_module():
    mod = pytest.importorskip(
        "spectramr.infrastructure.reporting.metrics_report_generator",
    )
    public = [a for a in dir(mod) if not a.startswith("_")]
    assert public, "metrics_report_generator module must expose a public API"
