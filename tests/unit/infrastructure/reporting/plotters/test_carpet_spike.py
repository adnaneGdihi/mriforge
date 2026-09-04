"""Tests for the QC carpet + spike diagnostics plotter.

Targets ``spectramr.infrastructure.reporting.plotters.qc.carpet_spike``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting import plotters


def _cases(n=5):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        pred = rng.random((12, 20)).astype(np.float32)
        out.append({
            "prediction": pred,
            "target": pred + 0.05 * rng.random((12, 20)).astype(np.float32),
            "metrics": {"psnr": 30.0},
            "rank": "case",
            "case_id": f"val_step{i}",
        })
    return out


def test_registered():
    assert "qc_carpet" in plotters.list_available()


def test_renders_carpet(tmp_path):
    res = plotters.get("qc_carpet")(pd.DataFrame(), tmp_path / "c.pdf",
                                       cases=_cases(), formats=("pdf",))
    assert res is not None and res.exists()


def test_soft_skip_when_no_cases(tmp_path):
    res = plotters.get("qc_carpet")(pd.DataFrame(), tmp_path / "c.pdf",
                                       cases=[], formats=("pdf",))
    assert res is None


def test_handles_case_without_target(tmp_path):
    cases = _cases(2)
    for c in cases:
        c.pop("target", None)
    res = plotters.get("qc_carpet")(pd.DataFrame(), tmp_path / "c.pdf",
                                       cases=cases, formats=("pdf",))
    assert res is not None and res.exists()
