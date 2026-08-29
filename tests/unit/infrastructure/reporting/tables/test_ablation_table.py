"""Render + empty-input contract tests for the ablation table (Tab 2.2).

Pins the "total function on bad input" contract: an empty frame, or one with
no ``full`` reference method, must return ``None`` rather than raise.
"""

import numpy as np
import pandas as pd

from mriforge.infrastructure.reporting.tables.ablation_table import make


def _cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method, base in (("full", 35.0), ("no_attn", 33.0), ("no_dc", 31.0)):
        for subj in range(3):
            rows.append({"method": method, "metric": "psnr", "subject_id": subj,
                         "value": base + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def test_ablation_table_renders_triple(tmp_path):
    out = make(_cohort(), tmp_path / "abl", primary_metric="psnr",
               full_method="full")
    assert out is not None
    for key in ("markdown", "latex", "csv"):
        assert out[key].exists()


def test_ablation_table_empty_df_returns_none(tmp_path):
    # Regression: empty frame has no ``metric`` column — guard must precede the
    # ``df["metric"] == primary_metric`` subscript (else KeyError, not None).
    out = make(pd.DataFrame(), tmp_path / "empty")
    assert out is None


def test_ablation_table_missing_full_method_returns_none(tmp_path):
    df = pd.DataFrame([{"method": "A", "metric": "psnr", "value": 30.0}])
    out = make(df, tmp_path / "no_full", primary_metric="psnr", full_method="full")
    assert out is None
