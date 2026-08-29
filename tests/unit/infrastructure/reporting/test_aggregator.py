"""Tests for the multi-arm reporting aggregator.

The task description points at ``src/infrastructure/reporting/aggregator.py``,
which is not present in this snapshot of the repo. The closest extant
abstraction is :class:`ComparativeReport.load_experiment_data`, which collects
per-arm result dicts and aggregates them into ``experiment_data``.

These tests cover that aggregation surface (multi-arm merge, empty inputs and
inconsistent metric keys) and ``pytest.importorskip`` the future
``aggregator`` module so the suite remains forward-compatible.
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


def _write_arm_report(root: Path, name: str, payload: dict) -> Path:
    """Persist a per-arm training_report.json the way TrainingReport does."""

    arm_dir = root / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "training_report.json").write_text(json.dumps(payload))
    return arm_dir


def _arm_payload(name: str, *, psnr=None, ssim=None, g_loss=None, d_loss=None) -> dict:
    return {
        "experiment_name": name,
        "config": {"model_type": "pix2pix", "learning_rate": 2e-4, "batch_size": 4},
        "metrics_history": {
            "epoch": [0, 1, 2],
            "g_loss": [1.0, 0.9, 0.8],
            "d_loss": [0.5, 0.55, 0.6],
            "psnr": [20.0, 22.0, 24.0],
            "ssim": [0.7, 0.75, 0.8],
            "learning_rate": [2e-4] * 3,
            "gradient_norm_g": [1.0] * 3,
            "gradient_norm_d": [1.0] * 3,
        },
        "final_metrics": {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "psnr": psnr,
            "ssim": ssim,
        },
        "training_duration": "0:00:10",
    }


# --------------------------------------------------------------------------- #
# Multi-arm merge
# --------------------------------------------------------------------------- #


def test_aggregator_merges_three_arms(tmp_path):
    """ComparativeReport must collect results from N arms into experiment_data
    keyed by arm name."""

    arms = []
    for i, name in enumerate(["armA", "armB", "armC"]):
        arm_dir = _write_arm_report(
            tmp_path,
            name,
            _arm_payload(
                name,
                psnr=24.0 + i,
                ssim=0.80 + 0.01 * i,
                g_loss=0.5 - 0.05 * i,
                d_loss=0.4 + 0.05 * i,
            ),
        )
        arms.append((name, arm_dir))

    report = create_comparative_report(
        experiment_names=[n for n, _ in arms],
        experiment_dirs=[p for _, p in arms],
        output_dir=tmp_path / "out",
    )

    assert isinstance(report, ComparativeReport)
    # Per-arm structure preserved.
    assert set(report.experiment_data.keys()) == {"armA", "armB", "armC"}
    # Per-metric structure preserved within each arm.
    for arm_name in ("armA", "armB", "armC"):
        data = report.experiment_data[arm_name]
        assert "final_metrics" in data
        assert "metrics_history" in data
        assert set(data["final_metrics"].keys()) >= {"psnr", "ssim", "g_loss", "d_loss"}


# --------------------------------------------------------------------------- #
# Empty inputs
# --------------------------------------------------------------------------- #


def test_aggregator_handles_empty_arm_list(tmp_path):
    """Calling the aggregator with no arms must yield an empty experiment_data
    mapping and a documented sentinel from the textual report."""

    report = create_comparative_report(
        experiment_names=[], experiment_dirs=[], output_dir=tmp_path
    )

    assert report.experiment_data == {}
    assert report.generate_comparison_report() == "No experiment data loaded"


def test_aggregator_skips_arms_missing_training_report(tmp_path):
    """An arm directory that doesn't contain ``training_report.json`` must be
    silently skipped (no crash) rather than aborting the whole aggregation."""

    good = _write_arm_report(tmp_path, "good", _arm_payload("good", psnr=23.0))
    bad = tmp_path / "bad"
    bad.mkdir()  # no training_report.json inside

    report = create_comparative_report(
        experiment_names=["good", "bad"],
        experiment_dirs=[good, bad],
        output_dir=tmp_path / "out",
    )

    assert "good" in report.experiment_data
    assert "bad" not in report.experiment_data


# --------------------------------------------------------------------------- #
# Inconsistent metric keys across arms
# --------------------------------------------------------------------------- #


def test_aggregator_handles_inconsistent_metric_keys(tmp_path):
    """Some arms log PSNR, others don't. The aggregator must keep both and the
    comparison report must mark missing metrics as 'N/A' rather than crashing."""

    rich = _write_arm_report(
        tmp_path,
        "rich",
        _arm_payload("rich", psnr=25.0, ssim=0.82, g_loss=0.4, d_loss=0.5),
    )
    sparse_payload = _arm_payload("sparse")
    # Strip psnr/ssim from final_metrics to simulate an arm that didn't track them.
    sparse_payload["final_metrics"] = {"g_loss": 0.45, "d_loss": 0.55}
    sparse = _write_arm_report(tmp_path, "sparse", sparse_payload)

    report = create_comparative_report(
        experiment_names=["rich", "sparse"],
        experiment_dirs=[rich, sparse],
        output_dir=tmp_path / "out",
    )

    text = report.generate_comparison_report()

    assert "rich" in text
    assert "sparse" in text
    # Sparse arm is missing PSNR/SSIM. The textual report uses 'N/A' as the
    # documented placeholder for missing metric values.
    assert "N/A" in text


# --------------------------------------------------------------------------- #
# Forward compatibility
# --------------------------------------------------------------------------- #


def test_discover_artifacts_finds_report_cases_dir(tmp_path):
    """``discover_artifacts`` exposes the recorded ``report_cases/`` directory."""

    from mriforge.infrastructure.reporting.aggregator import discover_artifacts

    # No report_cases dir → None.
    arts = discover_artifacts(tmp_path)
    assert arts.report_cases_dir is None

    # With report_cases dir → discovered path.
    (tmp_path / "report_cases").mkdir()
    arts = discover_artifacts(tmp_path)
    assert arts.report_cases_dir == tmp_path / "report_cases"


# --------------------------------------------------------------------------- #
# final_metrics.json + run_summary.json folding (2026-07-01 data-sourcing fix)
# --------------------------------------------------------------------------- #


def test_aggregate_folds_final_metrics_json(tmp_path):
    """``final_metrics.json`` (the per-arm campaign contract) becomes
    ``split='best'`` / ``split='final'`` rows; the ``_best`` suffix is stripped
    because the split already encodes it. Regression: the aggregator used to
    look only for ``final_eval.json``, which the training loop never writes."""

    from mriforge.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text(json.dumps({
        "schema_version": "1",
        "best": {"ssim_best": 0.86, "psnr_best": 28.4, "loss_best": 0.0},
        "final_loss": 0.12,
        "early_stopping_best_value": 9.11,
        "early_stopping_best_iteration": 100,
    }))

    df = aggregate(tmp_path, method_name="armA")
    best = df[df["split"] == "best"].set_index("metric")["value"]
    assert best["ssim"] == pytest.approx(0.86)
    assert best["psnr"] == pytest.approx(28.4)
    final = df[df["split"] == "final"].set_index("metric")["value"]
    assert final["final_loss"] == pytest.approx(0.12)
    assert final["early_stopping_best_value"] == pytest.approx(9.11)


def test_aggregate_folds_run_summary_facts(tmp_path):
    """Run-level facts (params → params_m, throughput, wall-time) land as
    ``split='run'`` rows — this is what lets the headline Pareto (cost axis
    ``params_m``) and the computational profile fire on real runs."""

    from mriforge.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "run_summary.json").write_text(json.dumps({
        "run_id": "arm-20260701-abc",
        "model_params": 12_963_937,
        "iterations_per_sec": 5.767,
        "duration_sec": 120.0,
        "effective_batch": 4,
        "host": "compute-9-4",
    }))

    df = aggregate(tmp_path, method_name="armA")
    run = df[df["split"] == "run"].set_index("metric")["value"]
    assert run["params_m"] == pytest.approx(12.963937)
    assert run["iterations_per_sec"] == pytest.approx(5.767)
    assert run["duration_min"] == pytest.approx(2.0)
    assert run["effective_batch"] == pytest.approx(4.0)


def test_aggregate_accepts_run_summary_under_logs(tmp_path):
    """Historic layout kept ``run_summary.json`` under ``logs/`` — still found."""

    from mriforge.infrastructure.reporting.aggregator import discover_artifacts

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run_summary.json").write_text(json.dumps({"model_params": 1_000_000}))
    arts = discover_artifacts(tmp_path)
    assert arts.run_summary_json == logs / "run_summary.json"


def test_aggregate_survives_corrupt_summary_jsons(tmp_path, caplog):
    """A present-but-corrupt JSON logs a warning and contributes no rows —
    it must never crash the end-of-run report."""

    from mriforge.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text("{not json")
    (tmp_path / "run_summary.json").write_text("[1,")

    with caplog.at_level("WARNING"):
        df = aggregate(tmp_path, method_name="armA")
    assert df.empty
    assert "final_metrics.json unreadable" in caplog.text
    assert "run_summary.json unreadable" in caplog.text


def test_final_metrics_skips_non_numeric_values(tmp_path):
    """Strings / nulls / bools in final_metrics.json are skipped, not coerced."""

    from mriforge.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text(json.dumps({
        "best": {"ssim_best": 0.9, "note_best": "n/a", "flag_best": True},
        "final_loss": None,
    }))
    df = aggregate(tmp_path, method_name="armA")
    assert set(df[df["split"] == "best"]["metric"]) == {"ssim"}
    assert df[df["split"] == "final"].empty


def test_melt_drops_all_nan_metric_columns(tmp_path):
    """A metric column that is entirely empty (e.g. the val_loss / val_total_loss
    headers pre-seeded into the training CSV but never written) is dropped, so it
    never surfaces as a phantom all-NaN series in fig_1_2_learning_curves."""
    import pandas as pd

    from mriforge.infrastructure.reporting.aggregator import _melt_metrics_csv

    csv = tmp_path / "training_metrics.csv"
    pd.DataFrame({
        "iteration": [0, 1, 2],
        "g_total_loss": [0.5, 0.4, 0.3],
        "val_loss": [None, None, None],  # phantom header, never populated
    }).to_csv(csv, index=False)

    melted = _melt_metrics_csv(csv, "train", "r", "m")
    metrics = set(melted["metric"].unique())
    assert "g_total_loss" in metrics
    assert "val_loss" not in metrics


@pytest.mark.xfail(
    reason="Worker-authored test; assumes aggregate() accepts a list, but the "
    "real API takes a path argument (experiment_dir). API contract mismatch.",
    strict=False,
)
def test_optional_future_aggregator_module(tmp_path):
    """Smoke check that any future ``aggregator`` module exposes an ``aggregate``
    entry-point and that ``aggregate([])`` returns a falsy empty aggregate."""

    aggregator = pytest.importorskip("mriforge.infrastructure.reporting.aggregator")
    aggregate = getattr(aggregator, "aggregate", None)
    if aggregate is None:
        pytest.skip("aggregator module exists but has no aggregate() entry-point")

    result = aggregate([])
    # Documented "empty aggregate" — either an empty container or None.
    assert (result is None) or (hasattr(result, "__len__") and len(result) == 0)
