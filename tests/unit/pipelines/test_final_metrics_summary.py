"""Unit tests for the final_metrics.json best-value summariser.

The campaign aggregator reads ``final_metrics.json`` from each arm's
output directory; this test fixes the schema and the
"higher-is-better" / "lower-is-better" classification logic.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mriforge.pipelines.train import _summarise_best_metrics_from_csv


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_returns_empty_when_csv_missing(tmp_path: Path) -> None:
    summary = _summarise_best_metrics_from_csv(str(tmp_path / "missing.csv"))
    assert summary == {}


def test_returns_empty_when_path_is_none() -> None:
    assert _summarise_best_metrics_from_csv(None) == {}


def test_psnr_takes_max(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "val_psnr": 25.0, "val_ssim": 0.80},
            {"iteration": 200, "epoch": 1, "val_psnr": 28.5, "val_ssim": 0.85},
            {"iteration": 300, "epoch": 2, "val_psnr": 27.0, "val_ssim": 0.83},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert summary["val_psnr_best"] == pytest.approx(28.5)
    assert summary["val_ssim_best"] == pytest.approx(0.85)


def test_loss_and_lpips_take_min(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "g_total_loss": 1.50, "val_lpips": 0.30},
            {"iteration": 200, "epoch": 1, "g_total_loss": 0.90, "val_lpips": 0.22},
            {"iteration": 300, "epoch": 2, "g_total_loss": 1.10, "val_lpips": 0.27},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert summary["g_total_loss_best"] == pytest.approx(0.90)
    assert summary["val_lpips_best"] == pytest.approx(0.22)


def test_skips_iteration_and_epoch_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [{"iteration": 100, "epoch": 0, "val_psnr": 25.0}],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert "iteration_best" not in summary
    assert "epoch_best" not in summary
    assert "val_psnr_best" in summary


def test_handles_empty_and_nan_cells(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "val_psnr": 25.0, "val_ssim": "nan"},
            {"iteration": 200, "epoch": 1, "val_psnr": "", "val_ssim": 0.85},
            {"iteration": 300, "epoch": 2, "val_psnr": 28.0, "val_ssim": 0.83},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert summary["val_psnr_best"] == pytest.approx(28.0)
    assert summary["val_ssim_best"] == pytest.approx(0.85)


def test_dice_and_iou_take_max(tmp_path: Path) -> None:
    """Higher-is-better classification covers segmentation metrics too."""
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "val_dice": 0.70, "val_iou": 0.55},
            {"iteration": 200, "epoch": 1, "val_dice": 0.82, "val_iou": 0.71},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))
    assert summary["val_dice_best"] == pytest.approx(0.82)
    assert summary["val_iou_best"] == pytest.approx(0.71)


def test_undeclared_metric_gets_no_best_entry(tmp_path: Path) -> None:
    """An undeclared column carries NO ``_best``, and is not guessed at.

    This used to assert "unknown -> default lower-is-better". That default was
    half of the defect `metric_directions` was built to end: the summariser's
    own 6-substring fallback (`psnr`/`ssim`/`accuracy`/`dice`/`f1`/`iou`) could
    only ever FLIP a column to maximize, and it fired exactly where the SSOT had
    declined to resolve. Defaulting the other way is the same mistake mirrored --
    a "best grad_norm" or "best lr" is not a quantity, and inventing one is what
    made `best_metric_name: lpips` maximize LPIPS.
    """
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "val_residual_norm": 5.0},
            {"iteration": 200, "epoch": 1, "val_residual_norm": 3.0},
            {"iteration": 300, "epoch": 2, "val_residual_norm": 7.0},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))

    assert "val_residual_norm_best" not in summary


def test_the_undeclared_column_is_named_not_silently_dropped(
    tmp_path: Path, caplog
) -> None:
    """Dropping in silence just moves the defect one layer.

    A reader looking for `val_residual_norm_best` finds nothing and cannot tell
    "no declared direction" from "that metric was never measured" -- the same
    ambiguity the not-applicable contract exists to remove.
    """
    import logging

    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [{"iteration": 100, "epoch": 0, "val_residual_norm": 5.0}],
    )
    with caplog.at_level(logging.INFO):
        _summarise_best_metrics_from_csv(str(csv_path))

    assert any("val_residual_norm" in r.getMessage() for r in caplog.records)


def test_a_declared_metric_is_unaffected(tmp_path: Path) -> None:
    """Anti-vacuity: the collection must not swallow declared columns."""
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            {"iteration": 100, "epoch": 0, "val_psnr": 20.0, "val_residual_norm": 5.0},
            {"iteration": 200, "epoch": 1, "val_psnr": 31.0, "val_residual_norm": 3.0},
        ],
    )
    summary = _summarise_best_metrics_from_csv(str(csv_path))

    assert summary["val_psnr_best"] == pytest.approx(31.0)
    assert "val_residual_norm_best" not in summary
