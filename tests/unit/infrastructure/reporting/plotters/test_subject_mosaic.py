"""Tests for the QC per-subject QC mosaic plotter.

Targets ``mriforge.infrastructure.reporting.plotters.qc.subject_mosaic``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting import plotters


def _cases(n=3):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        pred = rng.random((16, 16)).astype(np.float32)
        out.append({
            "prediction": pred,
            "target": pred + 0.1 * rng.random((16, 16)).astype(np.float32),
            "input": pred,
            "metrics": {"psnr": 30.0 + i, "ssim": 0.8},
            "rank": ["best", "median", "worst"][i % 3],
            "case_id": f"val_step{i}",
        })
    return out


def test_registered():
    assert "qc_subject_mosaic" in plotters.list_available()


def test_renders_mosaic(tmp_path):
    res = plotters.get("qc_subject_mosaic")(pd.DataFrame(), tmp_path / "m.pdf",
                                               cases=_cases(), formats=("pdf",))
    assert res is not None and res.exists()


def test_soft_skip_when_no_cases(tmp_path):
    res = plotters.get("qc_subject_mosaic")(pd.DataFrame(), tmp_path / "m.pdf",
                                               cases=None, formats=("pdf",))
    assert res is None


def test_handles_case_without_target(tmp_path):
    cases = _cases(1)
    del cases[0]["target"]
    res = plotters.get("qc_subject_mosaic")(pd.DataFrame(), tmp_path / "m.pdf",
                                               cases=cases, formats=("pdf",))
    assert res is not None and res.exists()
