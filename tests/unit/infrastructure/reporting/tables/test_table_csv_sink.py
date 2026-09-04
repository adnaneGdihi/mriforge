"""Tests for the machine-readable CSV sink on the publication tables.

Phase 3.1 of ``backlog_beautify_logging_and_reporting.md`` calls for
every table emitter to produce the full ``{tex, md, csv}`` triple. The
three publication tables (``main_results``, ``ablation_table``,
``hyperparameter_table``) render rich Markdown + LaTeX (bold
best-marking, mean±std cells, significance superscripts) that
``triple_emit`` cannot reproduce from a plain DataFrame, so they keep
their bespoke renderers and gain a raw-value CSV via
:func:`emit_table_csv`.

These tests pin: (a) the CSV file is written, (b) it is parseable raw
data (no LaTeX/markdown markup), (c) it round-trips the underlying
values.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.tables import (
    ablation_table,
    emit_table_csv,
    hyperparameter_table,
    main_results,
)


def _long_df() -> pd.DataFrame:
    """Minimal long-format aggregator frame: 2 methods x psnr/ssim x 3 seeds."""
    records = []
    for method, base in (("baseline", 24.0), ("ours", 26.0)):
        for seed in range(3):
            records.append(
                {"method": method, "metric": "psnr", "split": "test",
                 "value": base + seed * 0.1}
            )
            records.append(
                {"method": method, "metric": "ssim", "split": "test",
                 "value": 0.80 + seed * 0.01}
            )
    return pd.DataFrame(records)


def test_emit_table_csv_writes_parseable_file(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    path = emit_table_csv(df, tmp_path, "demo")
    assert path.exists() and path.suffix == ".csv"
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["a", "b"]
    assert reloaded.shape == (2, 2)


def test_main_results_emits_csv_sink(tmp_path) -> None:
    out = main_results.make(
        _long_df(), tmp_path, metrics=["psnr", "ssim"], baseline_method="baseline"
    )
    assert out is not None and "csv" in out
    csv = pd.read_csv(out["csv"])
    # Raw machine-readable columns, no markup.
    assert "method" in csv.columns
    assert "psnr_mean" in csv.columns and "psnr_std" in csv.columns
    # No LaTeX/markdown markup leaked into the CSV.
    raw = out["csv"].read_text()
    assert "textbf" not in raw and "**" not in raw and "±" not in raw


def test_ablation_emits_csv_sink(tmp_path) -> None:
    records = []
    for method, base in (("full", 26.0), ("no_attn", 25.0), ("no_skip", 24.0)):
        for seed in range(3):
            records.append(
                {"method": method, "metric": "psnr", "value": base + seed * 0.1}
            )
    out = ablation_table.make(pd.DataFrame(records), tmp_path, full_method="full")
    assert out is not None and "csv" in out
    csv = pd.read_csv(out["csv"])
    assert "variant" in csv.columns and "delta_vs_full" in csv.columns
    # The full model's delta must be exactly 0.
    full_row = csv[csv["variant"] == "full"].iloc[0]
    assert full_row["delta_vs_full"] == pytest.approx(0.0)


def test_hyperparameter_emits_csv_sink(tmp_path) -> None:
    hps = [
        {"name": "lr", "distribution": "log-uniform", "range": "[1e-5,1e-3]",
         "final": 2e-4, "criterion": "val_psnr"},
        {"name": "batch_size", "distribution": "choice", "range": "[2,4,8]",
         "final": 4, "criterion": "val_psnr"},
    ]
    out = hyperparameter_table.make(None, tmp_path, hyperparameters=hps)
    assert out is not None and "csv" in out
    csv = pd.read_csv(out["csv"])
    assert set(csv.columns) >= {"name", "distribution", "range", "final", "criterion"}
    assert csv.shape[0] == 2
