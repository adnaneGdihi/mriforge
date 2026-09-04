"""Tests for the interactive 2-D per-subject viewer.

Targets ``spectramr.infrastructure.reporting.interactive.viewer_2d``: builds a
scrubbable div from 2-D cases, soft-skips on empty input, and (with plotly) is
self-contained (no CDN fetch).
"""

from __future__ import annotations

import importlib.util

import numpy as np

from spectramr.infrastructure.reporting.interactive import viewer_2d as v2d

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def _cases(n=3):
    rng = np.random.default_rng(0)
    out = []
    for i, rank in enumerate(["best", "median", "worst"][:n]):
        tgt = rng.random((48, 48)).astype(np.float32)
        pred = (tgt + 0.05 * rng.random((48, 48))).astype(np.float32)
        out.append({"prediction": pred, "target": tgt, "rank": rank,
                    "case_id": f"v{i}", "metrics": {"psnr": 33.0 - i, "ssim": 0.9}})
    return out


def test_returns_none_on_empty():
    assert v2d.subject_viewer_2d_div([]) is None
    assert v2d.subject_viewer_2d_div(None) is None


def test_returns_none_when_no_prediction():
    assert v2d.subject_viewer_2d_div([{"target": np.zeros((8, 8), np.float32)}]) is None


def test_builds_scrubbable_div():
    div = v2d.subject_viewer_2d_div(_cases())
    if _HAS_PLOTLY:
        assert isinstance(div, str)
        assert "subject-2d" in div
        assert "How to read this" in div
        assert 'src="https://cdn' not in div
    else:
        assert div is None


def test_none_when_plotly_absent(monkeypatch):
    monkeypatch.setattr(v2d, "has_plotly", lambda: False)
    assert v2d.subject_viewer_2d_div(_cases()) is None


def test_no_global_rng_touch():
    before = repr(np.random.get_state())
    v2d.subject_viewer_2d_div(_cases())
    assert repr(np.random.get_state()) == before
