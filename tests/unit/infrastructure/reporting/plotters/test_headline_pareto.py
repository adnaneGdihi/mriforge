"""Render + empty-input contract tests for the headline-Pareto plotter.

The Pareto frontier maths (non-domination mask) is exercised by the render
test; here we also pin the "total function on bad input" contract — an empty
frame, or one missing the cost axis, must return ``None`` rather than raise.
"""

import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.plotters import headline_pareto


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method, params in (("full", 42.0), ("small", 8.0), ("big", 70.0)):
        rows.append({"method": method, "metric": "psnr", "step": float("nan"),
                     "value": 33.0 + rng.normal(0, 0.3), "split": "test"})
        rows.append({"method": method, "metric": "params_m", "step": float("nan"),
                     "value": params, "split": "meta"})
    return pd.DataFrame(rows)


def test_headline_pareto_renders(tmp_path):
    out = headline_pareto.make(_cohort(), tmp_path / "pareto.pdf",
                               metric="psnr", cost="params_m",
                               formats=("pdf",), dpi=150)
    assert out is not None and out.exists()


def test_headline_pareto_empty_df_returns_none(tmp_path):
    # Regression: empty frame has no ``metric`` column — guard must precede the
    # ``df["metric"].isin(...)`` subscript (else KeyError, not None).
    out = headline_pareto.make(pd.DataFrame(), tmp_path / "empty.pdf")
    assert out is None


def test_headline_pareto_missing_cost_axis_returns_none(tmp_path):
    # The cost metric was never logged: the pivot lacks that column and
    # ``dropna(subset=[metric, cost])`` would raise. Must return None instead.
    df = _cohort()
    df = df[df["metric"] == "psnr"].copy()  # drop the params_m rows
    out = headline_pareto.make(df, tmp_path / "no_cost.pdf",
                               metric="psnr", cost="params_m")
    assert out is None


def test_headline_pareto_fires_from_real_run_artifacts(tmp_path):
    """End-to-end through ``aggregate()``: with final_metrics.json (psnr) and
    run_summary.json (params_m) on disk the Pareto has both axes — the exact
    gap that kept this figure inert on real runs before 2026-07-01."""
    import json

    from mriforge.infrastructure.reporting.aggregator import aggregate_many

    dirs = []
    for name, params, psnr in (("ours", 12_000_000, 28.4),
                               ("baseline", 30_000_000, 26.1)):
        d = tmp_path / name
        d.mkdir()
        (d / "final_metrics.json").write_text(json.dumps(
            {"best": {"psnr_best": psnr}}))
        (d / "run_summary.json").write_text(json.dumps(
            {"model_params": params}))
        dirs.append(d)
    df = aggregate_many(dirs)
    out = headline_pareto.make(df, tmp_path / "pareto.pdf",
                               metric="psnr", cost="params_m",
                               formats=("pdf",), dpi=150)
    assert out is not None and out.exists()
