"""Render tests for the train/val generalization-gap figure (fig_1_18)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from mriforge.infrastructure.reporting.plotters import train_val_gap


def _frame() -> pd.DataFrame:
    """Mimic the real CSV layout: train logs ``train_ssim`` (plus stray
    ``val_*`` columns full of NaN), validation logs ``val_ssim`` sparsely."""
    rows = []
    for step in range(0, 100, 10):
        rows.append({"run_id": "r0", "method": "ours", "metric": "train_ssim",
                     "value": 0.5 + step / 250, "split": "train", "step": step})
        rows.append({"run_id": "r0", "method": "ours", "metric": "val_ssim",
                     "value": None, "split": "train", "step": step})
    for step in (20, 60, 90):
        rows.append({"run_id": "r0", "method": "ours", "metric": "val_ssim",
                     "value": 0.45 + step / 300, "split": "val", "step": step})
    return pd.DataFrame(rows)


def test_gap_matches_metrics_by_base_name(tmp_path):
    out = tmp_path / "gap.pdf"
    res = train_val_gap.make(_frame(), out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists() and res.stat().st_size > 0


def test_gap_returns_none_without_common_metric(tmp_path):
    df = _frame()
    df = df[df["split"] == "train"]
    assert train_val_gap.make(df, tmp_path / "gap.pdf") is None


def test_gap_total_on_empty_frame(tmp_path):
    assert train_val_gap.make(pd.DataFrame(), tmp_path / "gap.pdf") is None
