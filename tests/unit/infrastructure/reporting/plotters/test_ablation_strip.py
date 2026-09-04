"""Render + CI-wiring test for the ablation-strip plotter.

The CI math (Student-t, not z=1.96) is verified directly in
``test_stats.py::test_t_ci_half_width_*``; here we confirm the plotter wires
to that helper and renders for a small multi-seed cohort.
"""

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import ablation_strip


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method, base in (("full", 35.0), ("no_attn", 33.0), ("no_dc", 31.0)):
        for subj in range(3):
            rows.append(
                {
                    "method": method,
                    "metric": "psnr",
                    "subject_id": subj,
                    "value": base + rng.normal(0, 0.4),
                }
            )
    return pd.DataFrame(rows)


def test_ablation_strip_renders(tmp_path):
    out = tmp_path / "ablation_strip.pdf"
    res = ablation_strip.make(
        _cohort(), out, primary_metric="psnr", full_method="full",
        formats=("pdf",), dpi=150,
    )
    assert res is not None and res.exists()


def test_ablation_strip_empty_df_returns_none(tmp_path):
    # Regression: an empty frame has no ``metric`` column. The guard must
    # precede the ``df["metric"]`` subscript, else it raises KeyError instead
    # of returning the documented "no data" None.
    out = ablation_strip.make(pd.DataFrame(), tmp_path / "empty.pdf",
                              primary_metric="psnr", full_method="full")
    assert out is None


def test_ablation_strip_uses_student_t_helper(monkeypatch, tmp_path):
    # Wiring guard: the plotter must compute its CI through t_ci_half_width
    # (the fix), not an inline z=1.96 expression.
    calls = {"n": 0}
    import spectramr.infrastructure.reporting.plotters.ablation_strip as mod

    real = mod.t_ci_half_width

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "t_ci_half_width", spy)
    mod.make(_cohort(), tmp_path / "strip.pdf", primary_metric="psnr",
             full_method="full", formats=("pdf",), dpi=150)
    assert calls["n"] >= 1
