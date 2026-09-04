"""Render + CI-wiring test for the stratified-performance plotter.

The Student-t CI math is verified in ``test_stats.py``; here we confirm wiring
and rendering.
"""

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import stratified_performance


def _stratified() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for method in ("full", "ablate"):
        for group in ("easy", "hard"):
            for _ in range(3):
                rows.append(
                    {
                        "method": method,
                        "subgroup": group,
                        "metric": "psnr",
                        "value": (36 if group == "easy" else 30) + rng.normal(0, 0.5),
                    }
                )
    return pd.DataFrame(rows)


def test_stratified_performance_renders(tmp_path):
    out = tmp_path / "stratified.pdf"
    res = stratified_performance.make(
        pd.DataFrame(), out, stratified_df=_stratified(), metric="psnr",
        group_col="subgroup", formats=("pdf",), dpi=150,
    )
    assert res is not None and res.exists()


def test_stratified_performance_uses_student_t_helper(monkeypatch, tmp_path):
    import spectramr.infrastructure.reporting.plotters.stratified_performance as mod

    calls = {"n": 0}
    real = mod.t_ci_half_width

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "t_ci_half_width", spy)
    mod.make(pd.DataFrame(), tmp_path / "strat.pdf", stratified_df=_stratified(),
             metric="psnr", group_col="subgroup", formats=("pdf",), dpi=150)
    assert calls["n"] >= 1
