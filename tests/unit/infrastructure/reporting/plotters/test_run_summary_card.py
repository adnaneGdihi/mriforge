"""Render tests for the run-summary card (fig_1_16, 2026-07-01 upgrade)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from mriforge.infrastructure.reporting.plotters import run_summary_card


def _frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"run_id": "r0", "method": "ours", "metric": "params_m",
         "value": 12.96, "split": "run", "step": -1},
        {"run_id": "r0", "method": "ours", "metric": "iterations_per_sec",
         "value": 5.77, "split": "run", "step": -1},
        {"run_id": "r0", "method": "ours", "metric": "ssim",
         "value": 0.861, "split": "best", "step": -1},
        {"run_id": "r0", "method": "ours", "metric": "final_loss",
         "value": 0.12, "split": "final", "step": -1},
    ])


def test_card_renders_from_summary_rows(tmp_path):
    out = tmp_path / "card.pdf"
    res = run_summary_card.make(_frame(), out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists() and res.stat().st_size > 0


def test_card_returns_none_without_summary_rows(tmp_path):
    df = pd.DataFrame([
        {"run_id": "r0", "method": "ours", "metric": "loss",
         "value": 1.0, "split": "train", "step": 0},
    ])
    assert run_summary_card.make(df, tmp_path / "card.pdf") is None


def test_card_total_on_empty_frame(tmp_path):
    assert run_summary_card.make(pd.DataFrame(), tmp_path / "card.pdf") is None
