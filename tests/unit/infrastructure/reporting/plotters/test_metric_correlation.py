"""Render tests for the metric-correlation heatmap (fig_1_17, 2026-07-01)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import metric_correlation


def _frame(n_steps: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for step in range(n_steps):
        psnr = 20 + step + rng.normal(0, 0.1)
        rows += [
            {"run_id": "r0", "method": "ours", "metric": "val_psnr",
             "value": psnr, "split": "val", "step": step},
            # perfectly anti-correlated with psnr → strong off-diagonal signal
            {"run_id": "r0", "method": "ours", "metric": "val_nmse",
             "value": -psnr, "split": "val", "step": step},
            {"run_id": "r0", "method": "ours", "metric": "val_ssim",
             "value": rng.uniform(), "split": "val", "step": step},
        ]
    return pd.DataFrame(rows)


def test_correlation_heatmap_renders(tmp_path):
    out = tmp_path / "corr.pdf"
    res = metric_correlation.make(_frame(), out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists() and res.stat().st_size > 0


def test_correlation_needs_two_varying_metrics(tmp_path):
    df = _frame()
    df = df[df["metric"] == "val_psnr"]
    assert metric_correlation.make(df, tmp_path / "c.pdf") is None


def test_correlation_needs_min_steps(tmp_path):
    assert metric_correlation.make(_frame(n_steps=2), tmp_path / "c.pdf") is None


def test_correlation_drops_constant_series(tmp_path):
    # a constant metric has zero rank variance → dropped, not NaN-poisoned
    df = _frame()
    const = df[df["metric"] == "val_ssim"].copy()
    const["value"] = 1.0
    df = pd.concat([df[df["metric"] != "val_ssim"], const])
    out = tmp_path / "corr.pdf"
    res = metric_correlation.make(df, out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists()


def test_correlation_total_on_empty_frame(tmp_path):
    assert metric_correlation.make(pd.DataFrame(), tmp_path / "c.pdf") is None
