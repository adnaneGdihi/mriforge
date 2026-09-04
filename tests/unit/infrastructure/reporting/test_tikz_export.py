"""Tests for the TikZ/pgfplots exporter (2026-07-01 reporting upgrade).

Pure-text emission: writing ``.tex``/``.tikz`` must never require LaTeX.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import tikz_export


def _frame() -> pd.DataFrame:
    rows = []
    for method in ("ours", "baseline"):
        for step in range(0, 50, 10):
            rows.append({"run_id": method, "method": method, "metric": "loss",
                         "value": 1.0 / (1 + step), "split": "train", "step": step})
            rows.append({"run_id": method, "method": method, "metric": "val_psnr",
                         "value": 20.0 + step / 10, "split": "val", "step": step})
    # run-facts + best rows so the Pareto has both axes
    for method, params in (("ours", 12.9), ("baseline", 30.0)):
        rows.append({"run_id": method, "method": method, "metric": "params_m",
                     "value": params, "split": "run", "step": -1})
        rows.append({"run_id": method, "method": method, "metric": "psnr",
                     "value": 28.0 if method == "ours" else 26.5,
                     "split": "best", "step": -1})
    return pd.DataFrame(rows)


def test_curve_to_tikz_writes_standalone_and_snippet(tmp_path):
    tex = tikz_export.curve_to_tikz(_frame(), tmp_path, metric="loss",
                                    split="train", ymode="log")
    assert tex is not None and tex.exists()
    snippet = tex.with_suffix(".tikz")
    assert snippet.exists()
    body = tex.read_text()
    assert "\\documentclass[tikz]{standalone}" in body
    assert "ymode=log" in body
    assert "\\addplot" in body and "coordinates" in body
    # both methods present as legend entries
    assert "ours" in body and "baseline" in body
    # snippet is embeddable (no documentclass)
    assert "\\documentclass" not in snippet.read_text()


def test_curve_to_tikz_escapes_latex_specials():
    from spectramr.infrastructure.reporting.tikz_export import _tex_escape

    assert _tex_escape("val_ssim") == r"val\_ssim"
    assert _tex_escape("50%") == r"50\%"
    # single-pass escaping: the braces of \textbackslash{} stay intact
    assert _tex_escape("a\\b") == r"a\textbackslash{}b"


def test_curve_to_tikz_unknown_ymode_raises():
    with pytest.raises(ValueError, match="ymode"):
        tikz_export.curve_to_tikz(_frame(), "/tmp/never", metric="loss",
                                  ymode="banana")


def test_curve_to_tikz_total_on_bad_input(tmp_path):
    assert tikz_export.curve_to_tikz(pd.DataFrame(), tmp_path, metric="loss") is None
    assert tikz_export.curve_to_tikz(_frame(), tmp_path, metric="nope") is None


def test_pareto_to_tikz_uses_run_facts_cost_axis(tmp_path):
    tex = tikz_export.pareto_to_tikz(_frame(), tmp_path, metric="psnr",
                                     cost="params_m")
    assert tex is not None and tex.exists()
    body = tex.read_text()
    assert "parameters (M)" in body  # pretty label on the cost axis
    assert "PSNR (dB)" in body
    assert body.count("only marks") == 2


def test_export_tikz_figures_bundle(tmp_path):
    written = tikz_export.export_tikz_figures(_frame(), tmp_path)
    assert "tikz_curve_loss" in written
    assert "tikz_curve_val_psnr" in written
    assert "tikz_pareto_psnr" in written
    for path in written.values():
        assert path.exists() and path.suffix == ".tex"


def test_export_tikz_figures_empty_frame(tmp_path):
    assert tikz_export.export_tikz_figures(pd.DataFrame(), tmp_path) == {}


def test_coords_drop_nan_and_inf():
    from spectramr.infrastructure.reporting.tikz_export import _coords

    s = pd.Series([1.0, np.nan, np.inf, 2.0], index=[0, 1, 2, 3])
    coords = _coords(s)
    assert coords == "(0,1) (3,2)"
