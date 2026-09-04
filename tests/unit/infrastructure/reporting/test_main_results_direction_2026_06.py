"""Regression: the main-results table used a local higher-is-better heuristic.

2026-06-11 infrastructure audit. ``main_results.make`` had its own substring
heuristic that defaulted unknown metrics to higher-is-better, so a lower-better
metric whose name lacked a hardcoded keyword (``cjv``, ``efc``,
``bland_altman_bias``, ``ghosting_ratio``, …) had its WORST run bolded as best.
Direction now comes from ``METRIC_HIGHER_IS_BETTER`` (the same SSOT the
pipelines layer uses).
"""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.tables import main_results


def _df():
    # cjv is lower-is-better in the SSOT but lacks a heuristic keyword.
    return pd.DataFrame(
        [
            {"method": "methodA", "metric": "cjv", "value": 0.100, "split": "test"},
            {"method": "methodB", "metric": "cjv", "value": 0.900, "split": "test"},
        ]
    )


def test_lower_better_metric_bolds_the_lower_value(tmp_path):
    out = main_results.make(_df(), tmp_path, metrics=["cjv"])
    assert out is not None
    md = (tmp_path / "tab_2_1_main_results.md").read_text()
    rows = {
        line.split("|")[1].strip(): line
        for line in md.splitlines()
        if line.startswith("| method")
    }
    # methodA (0.100, the better/lower) must be the bolded best.
    assert "**0.100**" in rows["methodA"]
    # methodB (0.900, worse) must NOT be bolded as best.
    assert "**0.900**" not in rows["methodB"]
