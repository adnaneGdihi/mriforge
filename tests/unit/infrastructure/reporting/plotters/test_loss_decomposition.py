"""Render tests for the loss-decomposition stack (fig_1_3).

Paired with the 2026-07-01 clarity pass (pretty legend labels).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from mriforge.infrastructure.reporting.plotters import loss_decomposition


def _frame() -> pd.DataFrame:
    # component metrics must contain the component_keyword ("loss") in their
    # names — that is how the plotter tells components from other metrics
    rows = []
    for step in range(0, 50, 10):
        for metric, val in (("l1_loss", 0.5), ("adversarial_loss", 0.2),
                            ("loss", 0.7)):
            rows.append({"run_id": "r0", "method": "ours", "metric": metric,
                         "value": val / (1 + step / 10), "split": "train",
                         "step": step})
    return pd.DataFrame(rows)


def test_loss_decomposition_renders(tmp_path):
    out = tmp_path / "decomp.pdf"
    res = loss_decomposition.make(_frame(), out, formats=("pdf",), dpi=150)
    assert res is not None and res.exists() and res.stat().st_size > 0


def test_loss_decomposition_total_on_empty_frame(tmp_path):
    assert loss_decomposition.make(pd.DataFrame(), tmp_path / "d.pdf") is None


def test_loss_decomposition_none_without_components(tmp_path):
    df = _frame()
    df = df[df["metric"] == "loss"]  # total only, nothing to stack
    assert loss_decomposition.make(df, tmp_path / "d.pdf") is None
