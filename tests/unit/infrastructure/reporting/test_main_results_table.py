"""Tests for the main results table renderer.

The task description refers to ``src/infrastructure/reporting/tables/main_results.py``
which is not present in the current repo. The closest extant table renderer is
:meth:`ComparativeReport.generate_comparison_report`, which emits a textual
results table over multiple arms with per-metric rows.

Tests verify:
  * the table has the documented header,
  * one stanza per metric appears with one line per arm,
  * an empty results set produces the documented placeholder string,
  * a forward-compatible importorskip for any future tables.main_results module.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from mriforge.infrastructure.reporting.advanced_reporting import (
    ComparativeReport,
    create_comparative_report,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_arm(root: Path, name: str, psnr: float, ssim: float) -> Path:
    arm_dir = root / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": name,
        "config": {"model_type": "pix2pix"},
        "metrics_history": {
            "epoch": [0],
            "g_loss": [1.0],
            "d_loss": [0.5],
            "psnr": [psnr],
            "ssim": [ssim],
            "learning_rate": [2e-4],
            "gradient_norm_g": [1.0],
            "gradient_norm_d": [1.0],
        },
        "final_metrics": {
            "g_loss": 0.5,
            "d_loss": 0.4,
            "psnr": psnr,
            "ssim": ssim,
        },
        "training_duration": "0:00:10",
    }
    (arm_dir / "training_report.json").write_text(json.dumps(payload))
    return arm_dir


# --------------------------------------------------------------------------- #
# Table emits the expected header + N rows + per-metric alignment
# --------------------------------------------------------------------------- #


def test_main_results_table_renders_header_and_rows(tmp_path):
    arm_a = _write_arm(tmp_path, "armA", psnr=24.0, ssim=0.80)
    arm_b = _write_arm(tmp_path, "armB", psnr=25.5, ssim=0.82)
    arm_c = _write_arm(tmp_path, "armC", psnr=22.1, ssim=0.78)

    report = create_comparative_report(
        experiment_names=["armA", "armB", "armC"],
        experiment_dirs=[arm_a, arm_b, arm_c],
        output_dir=tmp_path / "out",
    )

    text = report.generate_comparison_report()

    # Header.
    assert "COMPARATIVE ANALYSIS REPORT" in text
    assert "EXPERIMENT SUMMARY" in text
    assert "DETAILED COMPARISON" in text

    # One stanza per metric, each containing one line per arm.
    metric_stanzas = (
        "PSNR Comparison",
        "SSIM Comparison",
        "G_LOSS Comparison",
        "D_LOSS Comparison",
    )
    for metric in metric_stanzas:
        assert metric in text, f"missing metric stanza: {metric}"

    # Each arm appears in the detailed comparison.
    detailed = text.split("DETAILED COMPARISON", 1)[1]
    for arm in ("armA", "armB", "armC"):
        assert arm in detailed


def test_main_results_table_formats_values_to_four_decimals(tmp_path):
    """The detailed-comparison rows render metric values with four-decimal
    precision, which is the documented alignment for the textual table."""

    arm_a = _write_arm(tmp_path, "armA", psnr=24.0, ssim=0.812345)
    report = create_comparative_report(
        experiment_names=["armA"],
        experiment_dirs=[arm_a],
        output_dir=tmp_path / "out",
    )

    text = report.generate_comparison_report()
    # 0.812345 → "0.8123" under the documented %.4f format.
    assert "0.8123" in text


# --------------------------------------------------------------------------- #
# Empty -> documented placeholder
# --------------------------------------------------------------------------- #


def test_main_results_table_empty_placeholder(tmp_path):
    report = ComparativeReport(experiment_names=[], output_dir=tmp_path)
    # No data loaded.
    assert report.generate_comparison_report() == "No experiment data loaded"


def test_main_results_table_save_writes_text_artifact(tmp_path):
    arm = _write_arm(tmp_path, "only", psnr=20.0, ssim=0.7)
    report = create_comparative_report(
        experiment_names=["only"],
        experiment_dirs=[arm],
        output_dir=tmp_path / "out",
    )

    # save_comparison_report writes the textual table to disk; plotting paths
    # are exercised but we don't assert on PNGs to keep the test deterministic.
    report.save_comparison_report()
    text_path = report.output_dir / "comparative_report.txt"
    assert text_path.exists()
    content = text_path.read_text()
    assert "only" in content
    assert "DETAILED COMPARISON" in content


# --------------------------------------------------------------------------- #
# Missing-metric rendering: N/A placeholder rather than KeyError
# --------------------------------------------------------------------------- #


def test_main_results_table_renders_NA_for_missing_metric(tmp_path):
    """If a per-arm payload lacks a final metric, the textual table must print
    'N/A' in its place rather than crashing."""

    arm_dir = tmp_path / "armX"
    arm_dir.mkdir()
    payload = {
        "experiment_name": "armX",
        "config": {"model_type": "pix2pix"},
        "metrics_history": {
            "epoch": [0],
            "g_loss": [1.0],
            "d_loss": [0.5],
            "psnr": [0.0],
            "ssim": [0.0],
            "learning_rate": [2e-4],
            "gradient_norm_g": [1.0],
            "gradient_norm_d": [1.0],
        },
        # Drop psnr/ssim from final_metrics entirely.
        "final_metrics": {"g_loss": 0.5, "d_loss": 0.4},
        "training_duration": "0:00:10",
    }
    (arm_dir / "training_report.json").write_text(json.dumps(payload))

    report = create_comparative_report(
        experiment_names=["armX"],
        experiment_dirs=[arm_dir],
        output_dir=tmp_path / "out",
    )

    text = report.generate_comparison_report()
    assert "armX" in text
    assert "N/A" in text


# --------------------------------------------------------------------------- #
# Forward compatibility
# --------------------------------------------------------------------------- #


def test_optional_future_main_results_table_module():
    mod = pytest.importorskip("mriforge.infrastructure.reporting.tables.main_results")
    public = [a for a in dir(mod) if not a.startswith("_")]
    assert public, "tables.main_results must expose a public API"
